import express, { Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';

const app = express();
const PORT = Number.parseInt(process.env.PORT || '3000', 10);

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
  const key = process.env.GEMINI_API_KEY;
  if (!key) return null;
  if (!aiClient) {
    aiClient = new GoogleGenAI({ apiKey: key });
  }
  return aiClient;
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
  res.json({ status: 'ok', time: new Date().toISOString() });
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
  const ai = getAI();

  if (ai) {
    try {
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: `${buildSystemPrompt()}\nНапиши короткую искреннюю запись в свой личный дневник (3-5 предложений) о сегодняшнем дне, мыслях о ${memory.user_name || 'близком человеке'} и домашней суете. Формат только текст дневника.`,
              },
            ],
          },
        ],
      });
      thought = response.text?.trim() || '';
    } catch (e) {
      console.warn('Gemini diary generation fallback:', e);
    }
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
  const ai = getAI();

  if (ai) {
    try {
      const contents: any[] = [];
      const sysPrompt = buildSystemPrompt();

      // Convert conversation history
      if (Array.isArray(history)) {
        for (const h of history.slice(-6)) {
          contents.push({
            role: h.sender === 'user' ? 'user' : 'model',
            parts: [{ text: h.text }],
          });
        }
      }

      contents.push({
        role: 'user',
        parts: [{ text: cleanText }],
      });

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents,
        config: {
          systemInstruction: sysPrompt,
          temperature: 0.85,
        },
      });

      reply = response.text?.trim() || '';
    } catch (err) {
      console.warn('Gemini chat error, switching to persona fallback:', err);
    }
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
  });
}

startServer();
