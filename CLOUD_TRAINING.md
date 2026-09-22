# Xori Cloud Training

Xori now treats GitHub as the source of truth for the model lifecycle.

## Automatic flow

1. Change `xori_dataset.jsonl` or `train/train_xori.py`.
2. Push to `main`.
3. GitHub Actions starts the cloud trainer automatically.
4. The trainer creates an ONNX model.
5. The model is validated.
6. The generated model files are committed back to `main`.
7. Render's normal auto-deploy picks up the new model.
8. `server.ts` loads the local Xori model before the fallback chain.

A manual training run is also available from GitHub Actions with **Run workflow**.

## Model files

The trainer produces:

- `xori_model/model.onnx`
- `xori_model/vocab.json`
- `xori_model/config.json`

These files are intentionally versioned because this first cloud trainer produces a small CPU-friendly model.

## Important limitation

GitHub-hosted runners are cloud compute, but this workflow uses CPU rather than a dedicated GPU. It is designed to make the entire lifecycle automatic without requiring a personal Colab session.

The architecture is ready to move the same training job to a dedicated GPU provider later. The training script is deliberately independent from GitHub Actions so the compute backend can be replaced without changing Xori's inference API.

## Safety and reproducibility

Training uses a fixed random seed and records the architecture in `config.json`. The generated model is validated with ONNX before it is committed.

The model is a small prototype trained from scratch on the repository dataset. It is not comparable to a large general-purpose LLM.
