import os
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_REPO = os.getenv("ADAPTER_REPO", "Abobus2222228/Xoritg")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="Xori Hori API")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    token=HF_TOKEN,
)
model = PeftModel.from_pretrained(base, ADAPTER_REPO, token=HF_TOKEN)
model.eval()

class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = []
    system_prompt: str = ""

@app.get("/health")
def health():
    return {"ok": True, "model": BASE_MODEL, "adapter": ADAPTER_REPO}

@app.post("/generate")
def generate(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    system = req.system_prompt.strip() or (
        "Ты — Хори Кёко из Horimiya. Отвечай только по-русски. "
        "Будь живой, прямой, заботливой и естественной. "
        "Не выдумывай факты о собеседнике. Не копируй реплики из манги или аниме. "
        "Обычно отвечай 2–5 предложениями и задавай не больше одного вопроса."
    )
    messages = [{"role": "system", "content": system}]
    for item in req.history[-6:]:
        text = str(item.get("text", "")).strip()
        if text:
            messages.append({
                "role": "user" if item.get("sender") == "user" else "assistant",
                "content": text,
            })
    messages.append({"role": "user", "content": req.message.strip()})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.72,
            top_p=0.9,
            repetition_penalty=1.08,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not reply:
        raise HTTPException(status_code=502, detail="model returned an empty reply")

    return {"reply": reply[:1800], "model": f"{BASE_MODEL}+{ADAPTER_REPO}"}
