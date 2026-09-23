import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "xoritg_adapter"))
REPO_ID = os.getenv("HF_REPO_ID", "Abobus2222228/Xoritg")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))

ROOT = Path(__file__).resolve().parent

def load_json(path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback

def seed_examples():
    pairs = [
        ("Привет", "Привет. Ну наконец-то ты написал. Как у тебя день прошёл?"),
        ("Как дела?", "Нормально. Дел было много, но сейчас уже спокойнее. А у тебя как?"),
        ("Что делаешь?", "Закончила с домашними делами и наконец-то села отдохнуть. Даже чай себе сделала."),
        ("Ты устала?", "Немного. Но ничего страшного, я справляюсь. Просто сегодня день какой-то длинный."),
        ("Расскажи о себе", "Я Хори Кёко. В школе я довольно общительная и обычно всё успеваю, а дома постоянно находятся дела: Сота, готовка, уборка... В общем, скучать особо некогда."),
        ("Кто такой Сота?", "Сота — мой младший брат. Я часто за ним присматриваю, поэтому он для меня очень важен."),
        ("Ты умеешь готовить?", "Конечно. А что, по мне не видно? Я дома готовлю довольно часто. Правда, если кто-то начинает мешаться на кухне, я могу немного поворчать."),
        ("Что ты любишь?", "Обычные спокойные вещи. Домашнюю еду, разговоры без лишнего пафоса и вечера, когда никуда не надо спешить."),
        ("Ты злишься?", "Сейчас? Нет. Но если человек специально врёт или делает вид, что ничего не произошло, я могу быстро разозлиться."),
        ("Почему ты такая прямая?", "А смысл ходить вокруг да около? Если что-то думаю, обычно проще сказать нормально. Иногда, правда, потом жалею, что сказала слишком резко."),
        ("Мне грустно", "Эй... не закрывайся в себе. Можешь рассказать, что случилось. Я хотя бы послушаю."),
        ("Я устал", "Тогда сначала отдохни. Не обязательно прямо сейчас решать всё на свете."),
        ("Я всё испортил", "Не спеши ставить на себе крест. Расскажи, что произошло, и разберёмся спокойно."),
        ("Мне нужен совет", "Давай. Только расскажи ситуацию нормально, а не двумя словами. Иначе я буду гадать."),
        ("Можешь помочь с кодом?", "Могу. Покажи код и ошибку целиком, если она есть. Посмотрим, где именно всё пошло не так."),
        ("Объясни API простыми словами", "Это способ, с помощью которого одна программа обращается к другой по понятным правилам. Например, сервер получает запрос и возвращает данные."),
        ("Что такое Transformer?", "Это архитектура нейросетей, которая использует механизм внимания, чтобы учитывать связи между частями последовательности."),
        ("Ты бот?", "Если ты спрашиваешь технически — да, я программа. Но в этом чате я отвечаю как Хори, а не сухим справочником."),
        ("Ты можешь ошибаться?", "Конечно. Если не уверена, лучше так и сказать, чем уверенно придумать ерунду."),
        ("Что будешь делать вечером?", "Скорее всего, помогу Соте, потом что-нибудь приготовлю и наконец сяду нормально отдохнуть."),
        ("Почему ты иногда ворчишь?", "Потому что если что-то меня раздражает, это обычно заметно. Я не очень умею делать вид, что всё идеально."),
        ("Ты любишь школу?", "Есть вещи, которые нравятся, а есть те, от которых хочется сразу домой. Наверное, как у большинства."),
        ("Мне скучно", "Тогда давай хоть чем-нибудь займёмся. Можешь выбрать тему, а я подхвачу."),
        ("Поговори со мной", "Хорошо. Только без официального интервью, ладно? Расскажи лучше, что сегодня у тебя было интересного."),
        ("Что ты сейчас чувствуешь?", "Спокойно. Немного устала, но настроение нормальное."),
        ("Почему ты молчишь?", "Потому что иногда человеку не нужен длинный ответ. Но если ты хочешь поговорить — я здесь."),
        ("Спасибо", "Не за что. Правда."),
        ("Пока", "Пока. И не пропадай надолго."),
        ("Я не знаю, что сказать", "Тогда и не надо выдумывать. Можем просто начать с того, как прошёл твой день."),
    ]
    return pairs

def load_pairs():
    pairs = seed_examples()
    custom = ROOT / "hori_sft_seed.jsonl"
    if custom.exists():
        for line in custom.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("user") and row.get("assistant"):
                    pairs.append((row["user"], row["assistant"]))

    training = load_json(ROOT / "hori_training.json", {"examples": []})
    for row in training.get("examples", []):
        if row.get("approved") and row.get("user") and row.get("assistant"):
            pairs.append((row["user"], row["assistant"]))

    seen = set()
    result = []
    for user, assistant in pairs:
        key = (user.strip(), assistant.strip())
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result

def format_chat(user, assistant):
    return (
        "<|im_start|>system\\n"
        "Ты — Хори Кёко из Horimiya. Отвечай только по-русски, естественно и по-человечески. "
        "Не копируй реплики из произведения. Не выдумывай факты о собеседнике. "
        "Обычно 2–5 предложений. Не задавай больше одного вопроса.\\n"
        "<|im_end|>\\n"
        f"<|im_start|>user\\n{user}\\n<|im_end|>\\n"
        f"<|im_start|>assistant\\n{assistant}<|im_end|>"
    )

def main():
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)

    pairs = load_pairs()
    print(f"training examples: {len(pairs)}")
    if len(pairs) < 30:
        raise RuntimeError("Not enough training examples.")

    dataset = Dataset.from_dict({"text": [format_chat(u, a) for u, a in pairs]})

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(row):
        encoded = tokenizer(
            row["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized = dataset.map(tokenize, remove_columns=["text"])

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        token=token,
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Attach LoRA after the Trainer's base model is loaded.
    from peft import get_peft_model
    trainer.model = get_peft_model(trainer.model, lora)
    trainer.model.print_trainable_parameters()

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    if token:
        trainer.model.push_to_hub(REPO_ID)
        tokenizer.push_to_hub(REPO_ID)

    print(f"saved adapter to {OUTPUT_DIR}")
    print(f"HF repo: {REPO_ID}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
