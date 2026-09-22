import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("XORI_MODEL_ID", "Abobus2222228/Xoritg")
MAX_NEW_TOKENS = int(os.getenv("XORI_MAX_NEW_TOKENS", "256"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

def generate(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are Xori, an independent AI assistant. Speak naturally in the user's language, especially Russian when the user writes Russian. Be useful, coherent and honest. Do not claim to be human.",
        },
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.08,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Textbox(label="Response"),
    title="Xori",
    description="Remote conversation inference for the Xori project.",
).launch()
