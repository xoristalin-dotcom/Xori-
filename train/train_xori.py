import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "xori_dataset.jsonl"
OUT = ROOT / "xori_model"
OUT.mkdir(parents=True, exist_ok=True)

MAX_LEN = int(os.getenv("XORI_MAX_LEN", "256"))
D_MODEL = int(os.getenv("XORI_D_MODEL", "192"))
N_HEADS = int(os.getenv("XORI_N_HEADS", "6"))
N_LAYERS = int(os.getenv("XORI_N_LAYERS", "4"))
EPOCHS = int(os.getenv("XORI_EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("XORI_BATCH", "16"))
LR = float(os.getenv("XORI_LR", "0.0003"))
SEED = int(os.getenv("XORI_SEED", "42"))

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def load_texts():
    texts = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if text:
                texts.append(text)
    if len(texts) < 8:
        raise RuntimeError("xori_dataset.jsonl needs at least 8 training examples")
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
    return ids[:MAX_LEN]


def make_batch(texts, vocab, indices):
    pad = vocab["<pad>"]
    batch = torch.full((len(indices), MAX_LEN), pad, dtype=torch.long)
    for row, idx in enumerate(indices):
        seq = encode(texts[idx], vocab)
        batch[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


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
            norm_first=True,
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
        return self.lm_head(self.norm(x))


def evaluate(model, ids, loss_fn, vocab_size):
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(ids), BATCH_SIZE):
            batch = ids[start:start + BATCH_SIZE]
            logits = model(batch)
            loss = loss_fn(
                logits[:, :-1, :].reshape(-1, vocab_size),
                batch[:, 1:].reshape(-1),
            )
            total += float(loss)
            batches += 1
    return total / max(1, batches)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    texts = load_texts()
    vocab = build_vocab(texts)

    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = max(1, int(len(indices) * 0.9))
    train_idx = indices[:split]
    val_idx = indices[split:] or indices[-1:]

    train_ids = make_batch(texts, vocab, train_idx)
    val_ids = make_batch(texts, vocab, val_idx)

    model = XoriGPT(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    best_val = float("inf")
    best_state = None

    print(f"examples={len(texts)} train={len(train_idx)} val={len(val_idx)} vocab={len(vocab)}")
    print(f"parameters={sum(p.numel() for p in model.parameters())}")

    for epoch in range(EPOCHS):
        model.train()
        order = train_idx[:]
        random.shuffle(order)
        total = 0.0
        batches = 0

        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[start:start + BATCH_SIZE]
            x = make_batch(texts, vocab, batch_indices)
            logits = model(x)
            loss = loss_fn(
                logits[:, :-1, :].reshape(-1, len(vocab)),
                x[:, 1:].reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += float(loss.detach())
            batches += 1

        scheduler.step()
        train_loss = total / max(1, batches)
        val_loss = evaluate(model, val_ids, loss_fn, len(vocab))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch={epoch + 1}/{EPOCHS} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model_path = OUT / "model.onnx"
    example = train_ids[:1]

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
        "version": 3,
        "maxSeqLen": MAX_LEN,
        "bosId": vocab["<bos>"],
        "eosId": vocab["<eos>"],
        "unkId": vocab["<unk>"],
        "padId": vocab["<pad>"],
        "vocabSize": len(vocab),
        "dModel": D_MODEL,
        "nHeads": N_HEADS,
        "nLayers": N_LAYERS,
        "bestValLoss": best_val,
        "examples": len(texts),
        "trainedBy": "GitHub Actions cloud trainer",
    }
    with (OUT / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"exported {model_path} ({model_path.stat().st_size} bytes)")
    print(f"best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()

# Training pipeline v0.3.1: rerun after push-race fix.
