import 'dotenv/config';
import express, { Request, Response } from 'express';
import path from 'path';
import os from 'os';
import fs from 'fs';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';

function loadDotEnvFromProjectAndHome() {
  const candidates = [
    path.join(process.cwd(), '.env'),
    path.join(process.cwd(), '.env.local'),
    path.join(os.homedir(), '.config', 'hori-kyoko', '.env'),
    path.join(os.homedir(), '.hori-kyoko.env'),
  ];

  for (const filePath of candidates) {
    if (!fs.existsSync(filePath)) continue;

    const content = fs.readFileSync(filePath, 'utf-8');
    for (const rawLine of content.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith('#') || !line.includes('=')) continue;

      const [key, ...rest] = line.split('=');
      const value = rest.join('=').trim();
      if (!process.env[key] && value) {
        process.env[key] = value.replace(/^['"]|['"]$/g, '');
      }
    }
  }
}

loadDotEnvFromProjectAndHome();

const app = express();
const PORT = Number.parseInt(process.env.PORT || '3000', 10);
const OFFLINE_MODE = process.env.OFFLINE_MODE === 'true' || process.env.OFFLINE_MODE === '1';
const OLLAMA_BASE_URL = (process.env.OLLAMA_BASE_URL || 'http://127.0.0.1:11434').replace(/\/$/, '');
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'gemma4:31b';
const TELEGRAM_BOT_TOKEN = process.env.BOT_TOKEN || '';

app.use(express.json({ limit: '10mb' }));

// Paths to JSON data files
const BASE_DIR = process.cwd();
const MEMORY_PATH = path.join(BASE_DIR, 'hori_memory.json');
const DIARY_PATH = path.join(BASE_DIR, 'hori_diary.json');
const PERSONALITY_PATH = path.join(BASE_DIR, 'hori_personality.json');
const KNOWLEDGE_PATH = path.join(BASE_DIR, 'hori_knowledge.json');

// Helper to safely load JSON
function loadJson<T>(filePath: string, fallback: T): T {
  try {
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf-8');
      return JSON.parse(content);
    }
  } catch (err) {
    console.error(`Error loading JSON from ${filePath}:`, err);
  }
  return fallback;
}

// Helper to save JSON
function saveJson(filePath: string, data: any) {
  try {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
  } catch (err) {
    console.error(`Error saving JSON to ${filePath}:`, err);
  }
}

// Lazy Gemini AI initialization
let aiClient: GoogleGenAI | null = null;
function getAI(): GoogleGenAI | null {
  if (OFFLINE_MODE) return null;

  const key = process.env.GEMINI_API_KEY;
  if (!key) return null;
  if (!aiClient) {
    aiClient = new GoogleGenAI({ apiKey: key });
  }
  return aiClient;
}

async function callOpenAICompatible({
  baseUrl,
  apiKey,
  model,
  messages,
  temperature,
}: {
  baseUrl: string;
  apiKey: string;
  model: string;
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
  temperature?: number;
}): Promise<string> {
  const response = await fetch(`${baseUrl.replace(/\/$/, '')}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
      'HTTP-Referer': 'https://localhost',
      'X-Title': 'Hori Kyoko',
    },
    body: JSON.stringify({
      model,
      messages,
      temperature: temperature ?? 0.85,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Provider request failed (${response.status}): ${text.slice(0, 400)}`);
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content;

  if (typeof content === 'string') return content.trim();
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === 'string' ? part : part?.text || ''))
      .join('')
      .trim();
  }

  return '';
}

async function callOllama({
  systemPrompt,
  userText,
  history,
}: {
  systemPrompt: string;
  userText: string;
  history: any[];
}): Promise<string> {
  if (process.env.OLLAMA_API_KEY || OLLAMA_BASE_URL.includes('ollama.com')) {
    return callOpenAICompatible({
      baseUrl: OLLAMA_BASE_URL,
      apiKey: process.env.OLLAMA_API_KEY || '',
      model: OLLAMA_MODEL,
      messages: buildChatMessages(systemPrompt, userText, history),
      temperature: 0.85,
    });
  }

  const messages = buildChatMessages(systemPrompt, userText, history);
  const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      messages,
      stream: false,
      options: { temperature: 0.85 },
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Ollama request failed (${response.status}): ${text.slice(0, 300)}`);
  }

  const data = await response.json();
  return typeof data.message?.content === 'string' ? data.message.content.trim() : '';
}

function buildChatMessages(systemPrompt: string, userText: string, history: any[] = []) {
  const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
    { role: 'system', content: systemPrompt },
  ];

  for (const h of history.slice(-6)) {
    const text = typeof h?.text === 'string' ? h.text : '';
    if (!text) continue;
    messages.push({
      role: h?.sender === 'user' ? 'user' : 'assistant',
      content: text,
    });
  }

  messages.push({ role: 'user', content: userText });
  return messages;
}

async function generateTextWithConfiguredProvider(systemPrompt: string, userText: string, history: any[] = []) {
  try {
    const localText = await callOllama({ systemPrompt, userText, history });
    if (localText) return localText;
  } catch (err) {
    console.warn('Ollama is unavailable, trying configured cloud providers:', err);
  }

  const gemini = getAI();
  if (gemini) {
    try {
      const contents: any[] = [];
      for (const h of history.slice(-6)) {
        contents.push({
          role: h.sender === 'user' ? 'user' : 'model',
          parts: [{ text: h.text || '' }],
        });
      }
      contents.push({ role: 'user', parts: [{ text: userText }] });

      const response = await gemini.models.generateContent({
        model: 'gemini-2.5-flash',
        contents,
        config: {
          systemInstruction: systemPrompt,
          temperature: 0.85,
        },
      });

      const text = response.text?.trim();
      if (text) return text;
    } catch (err) {
      console.warn('Gemini call failed, trying other configured providers:', err);
    }
  }

  if (OFFLINE_MODE) return '';

  const providerConfigs = [
    {
      enabled: Boolean(process.env.KIE_API_KEY),
      apiKey: process.env.KIE_API_KEY,
      baseUrl: process.env.KIE_BASE_URL || 'https://api.kie.ai/v1',
      model: process.env.KIE_MODEL || 'gpt-4o-mini',
      label: 'Kie.ai',
    },
    {
      enabled: Boolean(process.env.OPENROUTER_API_KEY),
      apiKey: process.env.OPENROUTER_API_KEY,
      baseUrl: 'https://openrouter.ai/api/v1',
      model: process.env.OPENROUTER_MODEL || 'openai/gpt-4o-mini',
      label: 'OpenRouter',
    },
    {
      enabled: Boolean(process.env.OPENAI_API_KEY),
      apiKey: process.env.OPENAI_API_KEY,
      baseUrl: 'https://api.openai.com/v1',
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      label: 'OpenAI',
    },
    {
      enabled: Boolean(process.env.GROQ_API_KEY),
      apiKey: process.env.GROQ_API_KEY,
      baseUrl: 'https://api.groq.com/openai/v1',
      model: process.env.GROQ_MODEL || 'llama-3.1-8b-instant',
      label: 'Groq',
    },
  ];

  for (const provider of providerConfigs) {
    if (!provider.enabled || !provider.apiKey) continue;

    try {
      const text = await callOpenAICompatible({
        baseUrl: provider.baseUrl,
        apiKey: provider.apiKey,
        model: provider.model,
        messages: buildChatMessages(systemPrompt, userText, history),
        temperature: 0.85,
      });

      if (text) return text;
    } catch (err) {
      console.warn(`${provider.label} call failed:`, err);
    }
  }

  return '';
}

type TelegramUpdate = {
  update_id: number;
  message?: {
    chat?: { id: number };
    text?: string;
  };
};

async function telegramRequest(method: string, body: Record<string, unknown> = {}) {
  if (!TELEGRAM_BOT_TOKEN) return null;
  const response = await fetch(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(`Telegram ${method} failed: ${JSON.stringify(data).slice(0, 400)}`);
  }
  return data.result;
}

async function sendTelegramReply(chatId: number, text: string) {
  const chunks = text.match(/[\s\S]{1,4000}/g) || ['Я здесь, но не смогла сформулировать ответ.'];
  for (const chunk of chunks) {
    await telegramRequest('sendMessage', { chat_id: chatId, text: chunk });
  }
}

async function handleTelegramMessage(chatId: number, text: string, history: any[]) {
  const { cleanText, warning } = moderateInput(text);
  if (warning) return warning;
  if (!cleanText) return 'Напиши мне что-нибудь, мой любимый.';

  let reply = '';
  try {
    reply = await generateTextWithConfiguredProvider(buildSystemPrompt(), cleanText, history);
  } catch (err) {
    console.warn('Telegram AI error, switching to persona fallback:', err);
  }

  if (!reply) {
    reply = generatePersonaFallback(cleanText, detectEmotion(cleanText)).reply;
  }

  const memory = loadJson<any>(MEMORY_PATH, { facts: [] });
  memory.emotion = detectEmotion(cleanText);
  memory.last_interaction = new Date().toISOString();
  saveJson(MEMORY_PATH, memory);
  return reply;
}

async function startTelegramPolling() {
  if (!TELEGRAM_BOT_TOKEN) {
    console.warn('Telegram polling is disabled: BOT_TOKEN is not configured.');
    return;
  }

  const histories = new Map<number, Array<{ sender: 'user' | 'hori'; text: string }>>();
  let offset = 0;
  await telegramRequest('deleteWebhook', { drop_pending_updates: false });
  const bot = await telegramRequest('getMe');
  console.log(`Telegram polling enabled for @${bot?.username || 'bot'}`);

  const poll = async () => {
    try {
      const updates = (await telegramRequest('getUpdates', {
        offset,
        timeout: 25,
        allowed_updates: ['message'],
      })) as TelegramUpdate[];

      for (const update of updates || []) {
        offset = update.update_id + 1;
        const chatId = update.message?.chat?.id;
        const text = update.message?.text?.trim();
        if (!chatId || !text) continue;

        if (text === '/start') {
          await sendTelegramReply(chatId, 'Привет! Я Хори Кёко. Напиши мне что-нибудь, и я отвечу.');
          continue;
        }

        const history = histories.get(chatId) || [];
        await telegramRequest('sendChatAction', { chat_id: chatId, action: 'typing' });
        const reply = await handleTelegramMessage(chatId, text, history.slice(-8));
        await sendTelegramReply(chatId, reply);
        const nextHistory: Array<{ sender: 'user' | 'hori'; text: string }> = [
          ...history,
          { sender: 'user', text },
          { sender: 'hori', text: reply },
        ];
        histories.set(chatId, nextHistory.slice(-12));
      }
    } catch (err) {
      console.error('Telegram polling error:', err);
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }
    void poll();
  };

  void poll();
}

// Safety & Emotion Detection from safety.py
const BLOCKED_PATTERNS = [
  /\b(?:kill|убей|убить|суицид|самоубийств)\b/i,
  /\b(?:нацист|расист|террорист)\w*\b/i,
];

function moderateInput(text: string): { cleanText: string; warning: string | null } {
  const trimmed = (text || '').trim();
  if (!trimmed) return { cleanText: '', warning: null };
  if (BLOCKED_PATTERNS.some((p) => p.test(trimmed))) {
    return { cleanText: '', warning: 'Давай без опасных тем. Я не буду помогать с причинением вреда.' };
  }
  return { cleanText: trimmed.slice(0, 4000), warning: null };
}

function detectEmotion(text: string): 'calm' | 'happy' | 'thinking' | 'sad' | 'angry' {
  const lower = (text || '').toLowerCase();
  const groups = {
    angry: ['злюсь', 'бесит', 'ненавиж', 'разозл', 'дурак', 'достал', 'хватит'],
    sad: ['груст', 'плохо', 'одиноко', 'плачу', 'устал', 'тяжело', 'тоск'],
    happy: ['рад', 'счаст', 'класс', 'отлично', 'ура', 'люблю', 'мило', 'прекрасн'],
    thinking: ['почему', 'как думаешь', 'не знаю', 'думаю', 'интересно', 'объясни'],
  };

  for (const [emotion, words] of Object.entries(groups)) {
    if (words.some((w) => lower.includes(w))) {
      return emotion as any;
    }
  }
  return 'calm';
}

function detectAnimationIntent(text: string): 'idle' | 'wave' | 'dance' | 'happy' {
  const lower = text.toLowerCase();
  if (lower.includes('станцуй') || lower.includes('потанцуй') || lower.includes('танец') || lower.includes('танцевать')) {
    return 'dance';
  }
  if (lower.includes('помаши') || lower.includes('привет') || lower.includes('здравствуй') || lower.includes('ку-ку') || lower.includes('хай')) {
    return 'wave';
  }
  if (lower.includes('рад') || lower.includes('счастлив') || lower.includes('ура') || lower.includes('класс') || lower.includes('здорово') || lower.includes('люблю')) {
    return 'happy';
  }
  return 'idle';
}

// Build system prompt for Hori Kyoko based on knowledge and personality
function buildSystemPrompt(): string {
  const personality = loadJson<any>(PERSONALITY_PATH, {});
  const knowledge = loadJson<any>(KNOWLEDGE_PATH, {});
  const memory = loadJson<any>(MEMORY_PATH, { user_name: 'мой любимый', facts: [] });

  const knownFacts = (memory.facts || []).map((f: any) => `- ${f.text}`).join('\n');

  return `Ты — Хори Кёко (Hori Kyouko) из Horimiya.
Полное имя: Хори Кёко. Ученица старшей школы Катагири.
Твой характер: общительная, энергичная, заботливая, прямая, ответственная, с живым юмором, иногда вспыльчивая, когда волнуешься, но искренняя и верная.
В школе ты популярная и аккуратная, а дома заботишься о младшем брате Соте, готовишь, убираешь и носишь простую домашнюю одежду.
Собеседника ты называешь: "${memory.user_name || 'мой любимый'}".
Факты, которые ты помнишь о собеседнике:
${knownFacts || '(пока мало фактов, внимательно слушай собеседника)'}

Правила ответа:
1. Всегда отвечай строго на русском языке, от первого лица, как Хори Кёко.
2. Никогда не говори, что ты ИИ, бот или языковая модель.
3. Стиль: разговорный, живой, теплый, прямой. Обычный размер 2-5 предложений.
4. Если собеседник делится личным или просит что-то запомнить, покажи, что ты услышала и запомнила.
5. Не используй служебные теги в финальном тексте.`;
}

// Fallback in-character responses if GEMINI_API_KEY is not configured
function generatePersonaFallback(userMessage: string, emotion: string): { reply: string; animation: 'idle' | 'wave' | 'dance' | 'happy' } {
  const memory = loadJson<any>(MEMORY_PATH, { user_name: 'мой любимый' });
  const name = memory.user_name || 'мой хороший';
  const lower = userMessage.toLowerCase();

  if (lower.includes('привет') || lower.includes('здравствуй') || lower.includes('хай')) {
    return {
      reply: `Привет-привет, ${name}! Как раз выдалась свободная минутка между делами. Как твои дела сегодня?`,
      animation: 'wave',
    };
  }

  if (lower.includes('танец') || lower.includes('станцуй') || lower.includes('потанцуй')) {
    return {
      reply: `Ой, ну ты и придумал! Хотя... ладно, только для тебя немного подвигаюсь. Смотри!`,
      animation: 'dance',
    };
  }

  if (lower.includes('готови') || lower.includes('ужин') || lower.includes('обед') || lower.includes('еда')) {
    return {
      reply: `Я как раз думала приготовить тушеное мясо с картошкой или карри для Соты. Ты голоден? Обязательно покушай нормально, не пропускай еду!`,
      animation: 'happy',
    };
  }

  if (lower.includes('миямур') || lower.includes('изуми')) {
    return {
      reply: `Миямура? Ну... он сначала казался таким мрачным в школе, а на самом деле такой заботливый и добрый. С ним спокойно, понимаешь?`,
      animation: 'idle',
    };
  }

  if (lower.includes('люблю') || lower.includes('нравишься')) {
    return {
      reply: `Ты... ты чего такое вдруг говоришь?! Прямо смутил меня... Но мне очень приятно это слышать, правда.`,
      animation: 'happy',
    };
  }

  if (emotion === 'sad') {
    return {
      reply: `Эй, что стряслось? Ты звучишь немного грустно. Расскажи мне, я всегда выслушаю. Не держи всё в себе.`,
      animation: 'idle',
    };
  }

  if (emotion === 'happy') {
    return {
      reply: `Вижу, у тебя отличное настроение! Это здорово, от твоей улыбки и у меня день становится светлее.`,
      animation: 'happy',
    };
  }

  return {
    reply: `Я тебя внимательно слушаю, ${name}! Расскажи побольше — мне всегда интересно, о чем ты думаешь.`,
    animation: 'idle',
  };
}

// ================= API ROUTES =================

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    time: new Date().toISOString(),
    offline: OFFLINE_MODE,
    localProvider: 'ollama',
    localModel: OLLAMA_MODEL,
    telegram: Boolean(TELEGRAM_BOT_TOKEN),
    hasGemini: Boolean(process.env.GEMINI_API_KEY),
  });
});

// Memory API
app.get('/api/memory', (req, res) => {
  const memory = loadJson(MEMORY_PATH, {
    user_name: 'мой любимый',
    facts: [],
    interests: [],
    conversations: [],
    mood: 'спокойное',
    emotion: 'calm',
    last_interaction: null,
    proactive_sent: [],
  });
  res.json(memory);
});

app.post('/api/memory/facts', (req, res) => {
  const { fact } = req.body;
  if (!fact || typeof fact !== 'string') {
    return res.status(400).json({ error: 'Fact text required' });
  }

  const memory = loadJson<any>(MEMORY_PATH, { facts: [] });
  if (!memory.facts) memory.facts = [];
  memory.facts.push({
    text: fact.trim(),
    time: new Date().toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' }),
  });
  saveJson(MEMORY_PATH, memory);
  res.json({ success: true, memory });
});

app.delete('/api/memory/facts/:index', (req, res) => {
  const index = parseInt(req.params.index, 10);
  const memory = loadJson<any>(MEMORY_PATH, { facts: [] });
  if (memory.facts && index >= 0 && index < memory.facts.length) {
    memory.facts.splice(index, 1);
    saveJson(MEMORY_PATH, memory);
  }
  res.json({ success: true, memory });
});

app.post('/api/memory/user_name', (req, res) => {
  const { user_name } = req.body;
  const memory = loadJson<any>(MEMORY_PATH, {});
  memory.user_name = (user_name || 'мой любимый').trim();
  saveJson(MEMORY_PATH, memory);
  res.json({ success: true, memory });
});

// Diary API
app.get('/api/diary', (req, res) => {
  const diary = loadJson(DIARY_PATH, { entries: [] });
  res.json(diary);
});

app.post('/api/diary', (req, res) => {
  const { title, text, mood, date, time } = req.body;
  if (!text) return res.status(400).json({ error: 'Entry text is required' });

  const diary = loadJson<any>(DIARY_PATH, { entries: [] });
  if (!diary.entries) diary.entries = [];

  const newEntry = {
    id: Date.now().toString(),
    title: title || 'Заметка Хори',
    text,
    mood: mood || 'Тёплое',
    date: date || new Date().toLocaleDateString('ru-RU'),
    time: time || new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
  };

  diary.entries.unshift(newEntry);
  saveJson(DIARY_PATH, diary);
  res.json({ success: true, entries: diary.entries });
});

app.post('/api/diary/generate', async (req, res) => {
  const memory = loadJson<any>(MEMORY_PATH, { user_name: 'мой любимый', facts: [] });
  const diary = loadJson<any>(DIARY_PATH, { entries: [] });
  if (!diary.entries) diary.entries = [];

  let thought = '';
  const promptText = `${buildSystemPrompt()}\nНапиши короткую искреннюю запись в свой личный дневник (3-5 предложений) о сегодняшнем дне, мыслях о ${memory.user_name || 'близком человеке'} и домашней суете. Формат только текст дневника.`;

  try {
    thought = await generateTextWithConfiguredProvider(promptText, promptText, []);
  } catch (e) {
    console.warn('Diary generation fallback:', e);
  }

  if (!thought) {
    const reflections = [
      `Сегодня был довольно насыщенный день. Сота снова попросил испечь печенье, а я поймала себя на мысли, как хорошо иногда просто остановиться и спокойно поболтать. Надеюсь, у ${memory.user_name || 'него'} тоже всё спокойно.`,
      `Вечером в доме Катагири наконец стало тихо. Закончила домашку и вымыла посуду. Приятно знать, что есть кто-то, кто всегда готов выслушать. Надо будет завтра приготовить что-нибудь вкусное.`,
      `Иногда бывает трудно совмещать школу и домашние заботы, но когда выдаётся минутка поговорить по душам — усталость как рукой снимает. Береги себя, ${memory.user_name || 'хороший мой'}.`,
    ];
    thought = reflections[Math.floor(Math.random() * reflections.length)];
  }

  const newEntry = {
    id: Date.now().toString(),
    title: `Вечерние мысли (${new Date().toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' })})`,
    text: thought,
    mood: 'Тёплое 🌸',
    date: new Date().toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' }),
    time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
  };

  diary.entries.unshift(newEntry);
  saveJson(DIARY_PATH, diary);
  res.json({ success: true, entries: diary.entries });
});

// Personality & Knowledge API
app.get('/api/personality', (req, res) => {
  const personality = loadJson(PERSONALITY_PATH, {});
  res.json(personality);
});

app.get('/api/knowledge', (req, res) => {
  const knowledge = loadJson(KNOWLEDGE_PATH, {});
  res.json(knowledge);
});

// Chat API with Gemini & Hori Persona
app.post('/api/chat', async (req, res) => {
  const { message, history } = req.body;

  const { cleanText, warning } = moderateInput(message);
  if (warning) {
    return res.json({
      reply: warning,
      emotion: 'thinking',
      animation: 'idle',
    });
  }

  const emotion = detectEmotion(cleanText);
  let animation = detectAnimationIntent(cleanText);

  // Extract simple facts automatically if user says "я люблю ...", "меня зовут ..."
  const memory = loadJson<any>(MEMORY_PATH, { facts: [] });
  if (!memory.facts) memory.facts = [];

  const lower = cleanText.toLowerCase();
  if (lower.startsWith('я люблю ') || lower.startsWith('мне нравится ') || lower.includes('мой любимый')) {
    const factText = cleanText.slice(0, 100);
    if (!memory.facts.some((f: any) => f.text === factText)) {
      memory.facts.push({
        text: factText,
        time: new Date().toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' }),
      });
      saveJson(MEMORY_PATH, memory);
    }
  }

  let reply = '';
  const sysPrompt = buildSystemPrompt();

  try {
    reply = await generateTextWithConfiguredProvider(sysPrompt, cleanText, Array.isArray(history) ? history : []);
  } catch (err) {
    console.warn('AI chat error, switching to persona fallback:', err);
  }

  // If Gemini is not set up or returned an empty response, use rich persona generator
  if (!reply) {
    const fallback = generatePersonaFallback(cleanText, emotion);
    reply = fallback.reply;
    if (animation === 'idle' && fallback.animation !== 'idle') {
      animation = fallback.animation;
    }
  }

  // Update memory state
  memory.emotion = emotion;
  memory.last_interaction = new Date().toISOString();
  saveJson(MEMORY_PATH, memory);

  res.json({
    reply,
    emotion,
    animation,
    memory,
  });
});

// Setup Vite middleware in dev or static serving in prod
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Хори Кёко 3D сервер запущен на http://0.0.0.0:${PORT}`);
    void startTelegramPolling().catch((err) => console.error('Telegram startup error:', err));
  });
}

startServer();
