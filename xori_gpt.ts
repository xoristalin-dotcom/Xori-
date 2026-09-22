import fs from 'node:fs';
import path from 'node:path';
import * as ort from 'onnxruntime-node';

const MODEL_DIR = path.join(process.cwd(), 'xori_model');
const MODEL_PATH = path.join(MODEL_DIR, 'model.onnx');
const VOCAB_PATH = path.join(MODEL_DIR, 'vocab.json');
const CONFIG_PATH = path.join(MODEL_DIR, 'config.json');

type ModelConfig = { version: number; maxSeqLen: number; bosId: number; eosId: number; unkId: number; };
let sessionPromise: Promise<ort.InferenceSession | null> | null = null;
let vocabCache: Record<string, number> | null = null;
let configCache: ModelConfig | null = null;

function loadAssets() {
  if (!fs.existsSync(MODEL_PATH) || !fs.existsSync(VOCAB_PATH) || !fs.existsSync(CONFIG_PATH)) return null;
  if (!vocabCache) vocabCache = JSON.parse(fs.readFileSync(VOCAB_PATH, 'utf8'));
  if (!configCache) configCache = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
  return { vocab: vocabCache, config: configCache };
}

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      if (!loadAssets()) return null;
      return ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['cpu'], graphOptimizationLevel: 'all' });
    })();
  }
  return sessionPromise;
}

function encode(text: string, vocab: Record<string, number>, unkId: number) {
  return Array.from(text).map(ch => vocab[ch] ?? unkId);
}

function decode(ids: number[], vocab: Record<string, number>) {
  const byId = new Map(Object.entries(vocab).map(([ch, id]) => [id, ch]));
  return ids.map(id => byId.get(id) ?? '').join('');
}

function argmax(values: Float32Array, offset: number, length: number) {
  let best = 0;
  for (let i = 1; i < length; i++) if (values[offset + i] > values[offset + best]) best = i;
  return best;
}

export async function generateXoriLocalReply(userText: string, history: Array<{ sender?: string; text?: string }> = []) {
  const assets = loadAssets();
  if (!assets) return null;
  const session = await getSession();
  if (!session) return null;
  const { vocab, config } = assets;
  const historyText = history.slice(-4).map(item => (item.sender === 'user' ? 'Пользователь: ' : 'Xori: ') + (item.text || '')).join('\n');
  const prompt = (historyText ? historyText + '\n' : '') + 'Пользователь: ' + userText + '\nXori:';
  let ids = [config.bosId, ...encode(prompt, vocab, config.unkId)].slice(-config.maxSeqLen + 1);

  for (let step = 0; step < 96; step++) {
    const input = new BigInt64Array(ids.map(id => BigInt(id)));
    const tensor = new ort.Tensor('int64', input, [1, ids.length]);
    const outputs = await session.run({ input_ids: tensor });
    const logits = outputs.logits;
    if (!logits || !(logits.data instanceof Float32Array)) return null;
    const vocabSize = logits.dims[2];
    const lastOffset = (ids.length - 1) * vocabSize;
    const nextId = argmax(logits.data as Float32Array, lastOffset, vocabSize);
    if (nextId === config.eosId || nextId === config.bosId) break;
    ids.push(nextId);
    if (ids.length >= config.maxSeqLen) break;
  }

  const generated = decode(ids.slice(1), vocab);
  const marker = generated.lastIndexOf('Xori:');
  const text = (marker >= 0 ? generated.slice(marker + 5) : generated).split('Пользователь:')[0].replace(/\s+/g, ' ').trim();
  if (text.length < 2) return null;
  return { text: text.slice(0, 1200), model: 'Xori Local GPT v0.1 (Colab-trained)' };
}

export function getXoriLocalModelStatus() {
  const assets = loadAssets();
  return { model: 'Xori Local GPT v0.1', trainedIn: 'Google Colab', format: 'ONNX', loaded: Boolean(assets), modelPath: MODEL_PATH, maxSeqLen: assets?.config.maxSeqLen ?? null };
}
