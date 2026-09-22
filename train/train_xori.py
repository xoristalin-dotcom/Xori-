import json
import os
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "xori_dataset.jsonl"
OUT = ROOT / "xori_model"
OUT.mkdir(parents=True, exist_ok=True)

MAX_LEN = int(os.getenv("XORI_MAX_LEN", "256"))
D_MODEL = int(os.getenv("XORI_D_MODEL", "128"))
N_HEADS = int(os.getenv("XORI_N_HEADS", "4"))
N_LAYERS = int(os.getenv("XORI_N_LAYERS", "4"))
EPOCHS = int(os.getenv("XORI_EPOCHS", "25"))
LR = float(os.getenv("XORI_LR", "0.0004"))

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def load_texts():
    texts = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if text:
                texts.append(text)
    if not texts:
        raise RuntimeError("xori_dataset.jsonl is empty")
    return texts


def build_vocab(texts):
    chars = sorted(set("".join(texts)))
    vocab = {token: i for i, token in enumerate(SPECIAL)}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode(text, vocab):
    unk = vocab["<unk>"]
    ids = [vocab["<bos>"]]
    ids.extend(vocab.get(ch, unk) for ch in text)
    ids.append(vocab["<eos>"])
    ids = ids[:MAX_LEN]
    ids += [vocab["<pad>"]] * (MAX_LEN - len(ids))
    return ids


class XoriGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, input_ids):
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos(positions)
        mask = torch.triu(
            torch.ones(length, length, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.encoder(x, mask=mask)
        return {"logits": self.lm_head(self.norm(x))}


def main():
    torch.manual_seed(42)
    texts = load_texts()
    vocab = build_vocab(texts)
    ids = torch.tensor([encode(t, vocab) for t in texts], dtype=torch.long)

    model = XoriGPT(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(ids))
        total = 0.0
        for idx in order:
            x = ids[idx : idx + 1]
            logits = model(x)["logits"]
            loss = loss_fn(logits[:, :-1, :].reshape(-1, len(vocab)), x[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch={epoch + 1}/{EPOCHS} loss={total / len(ids):.4f}")

    model.eval()
    model_path = OUT / "model.onnx"
    example = ids[:1]

    torch.onnx.export(
        model,
        (example,),
        str(model_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "sequence"},
            "logits": {1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
    )

    with (OUT / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    config = {
        "version": 2,
        "maxSeqLen": MAX_LEN,
        "bosId": vocab["<bos>"],
        "eosId": vocab["<eos>"],
        "unkId": vocab["<unk>"],
        "padId": vocab["<pad>"],
        "vocabSize": len(vocab),
        "dModel": D_MODEL,
        "nHeads": N_HEADS,
        "nLayers": N_LAYERS,
        "trainedBy": "GitHub Actions cloud trainer",
    }
    with (OUT / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"exported {model_path} ({model_path.stat().st_size} bytes)")
    print(f"vocab={len(vocab)} texts={len(texts)}")


if __name__ == "__main__":
    main()
