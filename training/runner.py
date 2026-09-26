import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

import requests
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = os.getenv("HF_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
CF_MODEL = os.getenv("CLOUDFLARE_LORA_MODEL", "@cf/meta/llama-3.2-3b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CF_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN") or os.getenv("CLOUDFLARE_AUTH_TOKEN")
RUNNER_TOKEN = os.getenv("TRAINING_RUNNER_TOKEN", "")

app = FastAPI(title="Xori GPU Training Runner")


class TrainRequest(BaseModel):
    candidateId: str
    manifest: dict
    datasetUrl: str
    callbackUrl: str


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for x in features:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def callback(url: str, candidate_id: str, status: str, runner=None, error=None, message=None):
    payload = {"candidateId": candidate_id, "status": status}
    if runner is not None:
        payload["runner"] = runner
    if error:
        payload["error"] = str(error)
    if message:
        payload["message"] = str(message)
    headers = {"Content-Type": "application/json"}
    if RUNNER_TOKEN:
        headers["Authorization"] = f"Bearer {RUNNER_TOKEN}"
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()


def download_dataset(url: str, workdir: Path) -> Path:
    out = workdir / "dataset.jsonl"
    headers = {}
    if RUNNER_TOKEN:
        headers["Authorization"] = f"Bearer {RUNNER_TOKEN}"
    with requests.get(url, headers=headers, timeout=60, stream=True) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out


def train_and_upload(req: TrainRequest):
    workdir = Path(tempfile.mkdtemp(prefix=f"xori-{req.candidateId}-"))
    try:
        callback(req.callbackUrl, req.candidateId, "training", message="GPU runner принял задачу.")
        dataset_path = download_dataset(req.datasetUrl, workdir)

        rows = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if len(rows) < 10:
            raise RuntimeError(f"Dataset too small: {len(rows)} examples")

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize(row):
            messages = row["messages"]
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": list(ids),
            }

        tokenized = [tokenize(row) for row in rows]
        split_at = max(1, int(len(tokenized) * 0.9))
        train_rows = tokenized[:split_at]
        eval_rows = tokenized[split_at:] or tokenized[-1:]

        from datasets import Dataset
        train_ds = Dataset.from_list(train_rows)
        eval_ds = Dataset.from_list(eval_rows)

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=HF_TOKEN,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

        lora = req.manifest.get("lora", {})
        config = LoraConfig(
            r=int(lora.get("r", 8)),
            lora_alpha=int(lora.get("alpha", 16)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora.get("targetModules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        )
        model = get_peft_model(model, config)

        output_dir = workdir / "adapter"
        training = req.manifest.get("training", {})
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=float(training.get("epochs", 4)),
            per_device_train_batch_size=int(training.get("batchSize", 1)),
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=int(training.get("gradientAccumulation", 16)),
            learning_rate=float(training.get("learningRate", 1e-4)),
            warmup_ratio=float(training.get("warmupRatio", 0.05)),
            weight_decay=float(training.get("weightDecay", 0.01)),
            fp16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="no",
            evaluation_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=Collator(tokenizer),
        )
        result = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        config_path = output_dir / "adapter_config.json"
        config_json = json.loads(config_path.read_text(encoding="utf-8"))
        config_json["model_type"] = "llama"
        config_path.write_text(json.dumps(config_json, indent=2) + "\n", encoding="utf-8")

        adapter_path = output_dir / "adapter_model.safetensors"
        if not adapter_path.exists() or not config_path.exists():
            raise RuntimeError("LoRA adapter files were not produced")
        if adapter_path.stat().st_size >= 300 * 1024 * 1024:
            raise RuntimeError("adapter_model.safetensors is >= 300 MB")

        if not CF_ACCOUNT_ID or not CF_TOKEN:
            raise RuntimeError("Cloudflare credentials are missing on the runner")

        name = f"xori-{req.candidateId}"
        create_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/finetunes"
        headers = {"Authorization": f"Bearer {CF_TOKEN}", "Content-Type": "application/json"}
        create = requests.post(
            create_url,
            headers=headers,
            json={
                "model": CF_MODEL,
                "name": name,
                "description": f"Xori Studio candidate {req.candidateId}",
                "public": False,
            },
            timeout=60,
        )
        create.raise_for_status()
        created = create.json()
        if not created.get("success") or not created.get("result", {}).get("id"):
            raise RuntimeError(f"Cloudflare create finetune failed: {created}")

        finetune_id = created["result"]["id"]
        upload_url = f"{create_url}/{finetune_id}/finetune-assets"

        for file_path in (config_path, adapter_path):
            with file_path.open("rb") as f:
                upload = requests.post(
                    upload_url,
                    headers={"Authorization": f"Bearer {CF_TOKEN}"},
                    files={"file": (file_path.name, f, "application/octet-stream")},
                    data={"file_name": file_path.name},
                    timeout=300,
                )
            upload.raise_for_status()
            body = upload.json()
            if not body.get("success"):
                raise RuntimeError(f"Cloudflare upload failed for {file_path.name}: {body}")

        callback(
            req.callbackUrl,
            req.candidateId,
            "trained",
            runner={
                "adapterId": finetune_id,
                "finetuneName": name,
                "model": CF_MODEL,
                "datasetSize": len(rows),
                "trainLoss": getattr(result, "training_loss", None),
            },
            message="LoRA обучена и загружена в Cloudflare. Candidate готов к тесту.",
        )
    except Exception as exc:
        try:
            callback(req.callbackUrl, req.candidateId, "runner_error", error=exc)
        except Exception:
            pass
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "baseModel": BASE_MODEL,
    }


@app.post("/train")
def train(req: TrainRequest, authorization: str | None = Header(default=None)):
    if RUNNER_TOKEN and authorization != f"Bearer {RUNNER_TOKEN}":
        raise HTTPException(status_code=401, detail="runner unauthorized")

    thread = threading.Thread(target=train_and_upload, args=(req,), daemon=True)
    thread.start()
    return {"ok": True, "status": "training_started", "candidateId": req.candidateId}
