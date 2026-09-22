import { pipeline, env } from '@huggingface/transformers';

const MODEL_ID = process.env.XORI_CONVERSATION_MODEL || 'onnx-community/Qwen2.5-0.5B-Instruct';
const DTYPE = process.env.XORI_CONVERSATION_DTYPE || 'q4';

let generatorPromise: Promise<any> | null = null;

env.allowLocalModels = false;
env.allowRemoteModels = true;

async function getGenerator() {
  if (!generatorPromise) {
    generatorPromise = pipeline('text-generation', MODEL_ID, {
      dtype: DTYPE as any,
      device: 'cpu',
    });
  }
  return generatorPromise;
}

function buildMessages(systemPrompt: string, userText: string, history: Array<{ sender?: string; text?: string }> = []) {
  const safeSystem = [
    'Ты Xori — самостоятельный AI-ассистент проекта Xori.',
    'Твоя главная задача — естественно и осмысленно общаться, понимать контекст и отвечать по существу.',
    'Отвечай на русском, если пользователь пишет по-русски.',
    'Не выдумывай факты, которых не знаешь. Если не уверен — прямо скажи об этом.',
    'Не повторяй вопрос пользователя без необходимости.',
    'Не говори, что ты человек или что у тебя есть реальная физическая жизнь.',
    'Не используй романтическую или сексуализированную ролевую манеру общения.',
    systemPrompt,
  ].filter(Boolean).join('\n');

  const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
    { role: 'system', content: safeSystem },
  ];

  for (const item of history.slice(-6)) {
    const text = String(item.text || '').trim();
    if (!text) continue;
    messages.push({
      role: item.sender === 'user' ? 'user' : 'assistant',
      content: text,
    });
  }

  messages.push({ role: 'user', content: userText });
  return messages;
}

export async function generateXoriConversationReply(
  systemPrompt: string,
  userText: string,
  history: Array<{ sender?: string; text?: string }> = [],
) {
  if (!userText.trim()) return null;

  const generator = await getGenerator();
  const messages = buildMessages(systemPrompt, userText, history);

  const result = await generator(messages, {
    max_new_tokens: Number(process.env.XORI_CONVERSATION_MAX_TOKENS || 192),
    temperature: Number(process.env.XORI_CONVERSATION_TEMPERATURE || 0.75),
    top_p: 0.9,
    repetition_penalty: 1.08,
    do_sample: true,
  });

  const chat = result?.[0]?.generated_text;
  const last = Array.isArray(chat) ? chat[chat.length - 1] : null;
  const text = typeof last?.content === 'string'
    ? last.content.trim()
    : typeof result?.[0]?.generated_text === 'string'
      ? result[0].generated_text.trim()
      : '';

  if (!text || text.length < 2) return null;

  return {
    text: text.slice(0, 3000),
    model: MODEL_ID,
    dtype: DTYPE,
  };
}

export function getXoriConversationModelStatus() {
  return {
    model: MODEL_ID,
    dtype: DTYPE,
    runtime: '@huggingface/transformers',
    device: 'cpu',
    role: 'primary conversational generator',
  };
}
