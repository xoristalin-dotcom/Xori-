import fs from 'node:fs';
import path from 'node:path';
import * as ort from 'onnxruntime-node';

const MODEL_DIR = path.join(process.cwd(), 'xori_model');
const MODEL_PATH = path.join(MODEL_DIR, 'model.onnx');
const VOCAB_PATH = path.join(MODEL_DIR, 'vocab.json');
const CONFIG_PATH = path.join(MODEL_DIR, 'config.json');

type ModelConfig = {
  version: number;
  maxSeqLen: number;
  bosId: number;
  eosId: number;
  unkId: number;
  padId?: number;
};

let sessionPromise: Promise<ort.InferenceSession | null> | null = null;
let vocabCache: Record<string, number> | null = null;
let configCache: ModelConfig | null = null;
let reverseVocabCache: Map<number, string> | null = null;

function loadAssets() {
  if (!fs.existsSync(MODEL_PATH) || !fs.existsSync(VOCAB_PATH) || !fs.existsSync(CONFIG_PATH)) return null;
  if (!vocabCache) {
    vocabCache = JSON.parse(fs.readFileSync(VOCAB_PATH, 'utf8'));
    reverseVocabCache = new Map(Object.entries(vocabCache).map(([token, id]) => [id, token]));
  }
  if (!configCache) configCache = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
  return { vocab: vocabCache, config: configCache };
}

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      if (!loadAssets()) return null;
      return ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all',
      });
    })().catch((error) => {
      console.error('[Xori Local GPT] ONNX load failed:', error);
      return null;
    });
  }
  return sessionPromise;
}

function encode(text: string, vocab: Record<string, number>, unkId: number) {
  return Array.from(text).map(ch => vocab[ch] ?? unkId);
}

function decode(ids: number[]) {
  if (!reverseVocabCache) return '';
  return ids.map(id => reverseVocabCache?.get(id) ?? '').join('');
}

function sampleFromLogits(
  values: Float32Array,
  offset: number,
  length: number,
  usedIds: number[],
  temperature = 0.75,
  topK = 24,
  repetitionPenalty = 1.08,
) {
  const candidates: Array<{ id: number; score: number }> = [];
  const recent = new Set(usedIds.slice(-48));

  for (let i = 0; i < length; i++) {
    let score = values[offset + i];
    if (recent.has(i)) score /= repetitionPenalty;
    candidates.push({ id: i, score });
  }

  candidates.sort((a, b) => b.score - a.score);
  const selected = candidates.slice(0, Math.min(topK, candidates.length));
  const scaled = selected.map(item => Math.exp((item.score - selected[0].score) / Math.max(0.2, temperature)));
  const total = scaled.reduce((sum, value) => sum + value, 0);
  let pick = Math.random() * total;

  for (let i = 0; i < selected.length; i++) {
    pick -= scaled[i];
    if (pick <= 0) return selected[i].id;
  }
  return selected[0].id;
}

export async function generateXoriLocalReply(
  userText: string,
  history: Array<{ sender?: string; text?: string }> = [],
) {
  const assets = loadAssets();
  if (!assets) return null;

  const session = await getSession();
  if (!session) return null;

  const { vocab, config } = assets;
  const historyText = history
    .slice(-6)
    .map(item => (item.sender === 'user' ? 'Пользователь: ' : 'Xori: ') + (item.text || ''))
    .join('\n');
  const prompt = (historyText ? historyText + '\n' : '') + 'Пользователь: ' + userText + '\nXori:';

  let ids = [config.bosId, ...encode(prompt, vocab, config.unkId)].slice(-config.maxSeqLen + 1);
  const promptLength = ids.length;

  // The exported Transformer currently has a fixed 256-token ONNX attention graph.
  // Keep the runtime input fixed-size and use only the real sequence prefix for
  // sampling. This prevents ONNX Runtime reshape failures on short prompts.
  const fixedLength = config.maxSeqLen;
  if (fixedLength < ids.length) ids = ids.slice(-fixedLength);

  for (let step = 0; step < 160 && ids.length < fixedLength; step++) {
    const padded = new BigInt64Array(fixedLength);
    padded.fill(BigInt(config.padId ?? 0));
    ids.forEach((id, index) => {
      padded[index] = BigInt(id);
    });

    const tensor = new ort.Tensor('int64', padded, [1, fixedLength]);
    const outputs = await session.run({ input_ids: tensor });
    const logits = outputs.logits;

    if (!logits || !(logits.data instanceof Float32Array)) return null;

    const vocabSize = logits.dims[2];
    const lastOffset = (ids.length - 1) * vocabSize;
    const nextId = sampleFromLogits(
      logits.data as Float32Array,
      lastOffset,
      vocabSize,
      ids,
    );

    if (nextId === config.eosId || nextId === config.bosId) break;
    ids.push(nextId);
    if (ids.length >= config.maxSeqLen) break;
  }

  const generated = decode(ids.slice(promptLength));
  const text = generated
    .split('Пользователь:')[0]
    .replace(/\s+/g, ' ')
    .trim();

  if (text.length < 2) return null;
  return {
    text: text.slice(0, 1200),
    model: 'Xori GPT v0.3 (from-scratch, cloud-trained)',
  };
}

export function getXoriLocalModelStatus() {
  const assets = loadAssets();
  return {
    model: 'Xori GPT v0.3',
    trainedIn: 'GitHub Actions cloud trainer',
    format: 'ONNX',
    loaded: Boolean(assets),
    modelPath: MODEL_PATH,
    maxSeqLen: assets?.config.maxSeqLen ?? null,
    vocabSize: assets?.config.vocabSize ?? null,
    examples: assets?.config.examples ?? null,
    validationLoss: assets?.config.bestValLoss ?? null,
  };
}

// Runtime fix: pad inference inputs to the exported fixed sequence length.
