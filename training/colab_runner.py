import os, json, time, tempfile, shutil
from pathlib import Path
import requests, torch

RENDER_URL = os.getenv("XORI_RENDER_URL", "https://xori-training-api.onrender.com")
TOKEN = os.environ["TRAINING_RUNNER_TOKEN"]

def run_once(job):
    workdir = Path(tempfile.mkdtemp(prefix="xori-colab-"))
    try:
        headers = {"Authorization": "Bearer " + TOKEN}
        dataset = workdir / "dataset.jsonl"
        with requests.get(job["datasetUrl"], headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            with dataset.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        rows = [json.loads(x) for x in dataset.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(rows) < 10:
            raise RuntimeError(f"Dataset too small: {len(rows)}")

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset

        manifest = job.get("manifest", {})
        base = manifest.get("baseModel", "meta-llama/Llama-3.2-3B-Instruct")
        hf_token = os.environ["HF_TOKEN"]
        tokenizer = AutoTokenizer.from_pretrained(base, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def encode(row):
            ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
            ids = ids.tolist() if hasattr(ids, "tolist") else ids
            return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids}

        data = [encode(x) for x in rows]
        cut = max(1, int(len(data) * 0.9))
        train_ds = Dataset.from_list(data[:cut])
        eval_ds = Dataset.from_list(data[cut:] or data[-1:])

        class Collator:
            def __call__(self, features):
                m = max(len(x["input_ids"]) for x in features)
                p = tokenizer.pad_token_id
                return {
                    "input_ids": torch.tensor([x["input_ids"] + [p] * (m-len(x["input_ids"])) for x in features]),
                    "attention_mask": torch.tensor([x["attention_mask"] + [0] * (m-len(x["attention_mask"])) for x in features]),
                    "labels": torch.tensor([x["labels"] + [-100] * (m-len(x["labels"])) for x in features]),
                }

        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(base, token=hf_token, quantization_config=q, device_map="auto", torch_dtype=torch.float16)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

        lora = manifest.get("lora", {})
        model = get_peft_model(model, LoraConfig(
            r=int(lora.get("r", 8)), lora_alpha=int(lora.get("alpha", 16)),
            lora_dropout=float(lora.get("dropout", 0.05)), bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora.get("targetModules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        ))

        tr = manifest.get("training", {})
        out = workdir / "adapter"
        args = TrainingArguments(
            output_dir=str(out), num_train_epochs=float(tr.get("epochs", 6)),
            per_device_train_batch_size=int(tr.get("batchSize", 1)),
            gradient_accumulation_steps=int(tr.get("gradientAccumulation", 16)),
            learning_rate=float(tr.get("learningRate", 1e-4)),
            warmup_ratio=float(tr.get("warmupRatio", 0.05)), weight_decay=float(tr.get("weightDecay", 0.01)),
            fp16=True, gradient_checkpointing=True, logging_steps=1, save_strategy="no",
            evaluation_strategy="epoch", report_to="none", remove_unused_columns=False
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=Collator())
        result = trainer.train()
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))

        config_path = out / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["model_type"] = "llama"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        adapter_path = out / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise RuntimeError("LoRA adapter was not produced")

        account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        cf_token = os.environ["CLOUDFLARE_API_TOKEN"]
        cf_model = "@cf/meta/llama-3.2-3b-instruct"
        create_url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/finetunes"
        cf_headers = {"Authorization": "Bearer " + cf_token, "Content-Type": "application/json"}
        cr = requests.post(create_url, headers=cf_headers, json={
            "model": cf_model, "name": "xori-" + job["candidateId"],
            "description": "Xori Studio candidate " + job["candidateId"], "public": False
        }, timeout=60)
        cr.raise_for_status()
        finetune_id = cr.json()["result"]["id"]

        for fp in (config_path, adapter_path):
            with fp.open("rb") as f:
                ur = requests.post(
                    f"{create_url}/{finetune_id}/finetune-assets",
                    headers={"Authorization": "Bearer " + cf_token},
                    files={"file": (fp.name, f, "application/octet-stream")},
                    data={"file_name": fp.name}, timeout=300
                )
            ur.raise_for_status()
            if not ur.json().get("success"):
                raise RuntimeError("Cloudflare upload failed")

        payload = {
            "candidateId": job["candidateId"], "status": "trained",
            "runner": {"adapterId": finetune_id, "finetuneName": "xori-" + job["candidateId"],
                       "model": cf_model, "datasetSize": len(rows),
                       "trainLoss": getattr(result, "training_loss", None),
                       "source": "colab", "adapterFormat": "PEFT LoRA", "baseModel": base},
            "message": "LoRA обучена в Colab и загружена в Cloudflare."
        }
        cb = requests.post(job["callbackUrl"], headers=headers, json=payload, timeout=60)
        cb.raise_for_status()
        print("Candidate trained:", job["candidateId"], "adapter:", finetune_id)
    except Exception as exc:
        try:
            requests.post(job["callbackUrl"], headers=headers, json={
                "candidateId": job["candidateId"], "status": "runner_error", "error": str(exc)[:2000]
            }, timeout=30)
        except Exception:
            pass
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

print("Xori Colab GPU runner started")
print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

while True:
    r = requests.get(RENDER_URL + "/api/studio/training/claim", headers={"Authorization": "Bearer " + TOKEN}, timeout=60)
    if r.status_code == 204:
        print("Queue empty; waiting 20s...")
        time.sleep(20)
        continue
    r.raise_for_status()
    job = r.json()
    print("Claimed:", job["candidateId"])
    run_once(job)
