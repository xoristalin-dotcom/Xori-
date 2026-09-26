import 'dotenv/config';
import express, { Request, Response } from 'express';
import path from 'path';
import os from 'os';
import fs from 'fs';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import { generateXoriNeuralReply, getXoriNeuralStatus } from './xori_neural';
import { generateXoriLocalReply, getXoriLocalModelStatus } from './xori_gpt';

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

const execFileAsync = promisify(execFile);

loadDotEnvFromProjectAndHome();

const app = express();
const PORT = Number.parseInt(process.env.PORT || '3000', 10);
const OFFLINE_MODE = process.env.OFFLINE_MODE === 'true' || process.env.OFFLINE_MODE === '1';
const TELEGRAM_BOT_TOKEN = process.env.BOT_TOKEN || process.env.TELEGRAM_BOT_TOKEN || '';
const TELEGRAM_VOICE_URL = process.env.TELEGRAM_VOICE_URL || '';
const TELEGRAM_PHOTO_URL = process.env.TELEGRAM_PHOTO_URL || '';
const POLLINATIONS_API_KEY = process.env.POLLINATIONS_API_KEY || '';
const HF_XORI_URL = process.env.HF_XORI_URL || '';
const HF_XORI_TOKEN = process.env.HF_XORI_TOKEN || process.env.HF_TOKEN || '';
const INTERNAL_LEARNING_ENABLED = process.env.INTERNAL_LEARNING_ENABLED !== 'false';
const WEB_SEARCH_ENABLED = process.env.WEB_SEARCH_ENABLED !== 'false';
const CONTROL_TOKEN = process.env.HORI_CONTROL_TOKEN || '';
const PROACTIVE_CHAT_ID = Number(process.env.HORI_PROACTIVE_CHAT_ID || 0);
let internalReplyCounter = 0;
const providerCooldowns = new Map<string, number>();
const providerLastErrors = new Map<string, string>();
const providerLastSuccess = new Map<string, string>();
const webSourceCooldowns = new Map<string, number>();
let telegramPollingStarted = false;

app.use(express.json({ limit: '10mb' }));

// Paths to JSON data files
const BASE_DIR = process.cwd();
const CONTROL_PATH = path.join(BASE_DIR, 'hori_control.json');
const MEMORY_PATH = path.join(BASE_DIR, 'hori_memory.json');
const DIARY_PATH = path.join(BASE_DIR, 'hori_diary.json');
const PERSONALITY_PATH = path.join(BASE_DIR, 'hori_personality.json');
const KNOWLEDGE_PATH = path.join(BASE_DIR, 'hori_knowledge.json');
const TRAINING_PATH = path.join(BASE_DIR, 'hori_training.json');

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
function getControlState() {
  return loadJson<any>(CONTROL_PATH, {
    enabled: true,
    proactiveEnabled: true,
    voiceEnabled: true,
    photoEnabled: true,
    autoFirst: true,
    modelMode: 'auto',
    updatedAt: null,
  });
}

function saveControlState(patch: any) {
  const current = getControlState();
  const next = {
    ...current,
    ...patch,
    updatedAt: new Date().toISOString(),
  };
  saveJson(CONTROL_PATH, next);
  return next;
}

function controlAuthorized(req: Request): boolean {
  if (!CONTROL_TOKEN) return false;
  return req.headers['x-hori-control-token'] === CONTROL_TOKEN;
}

function saveJson(filePath: string, data: any) {
  try {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
  } catch (err) {
    console.error(`Error saving JSON to ${filePath}:`, err);
  }
}

function tokenizeForLearning(text: string): Set<string> {
  const stopWords = new Set(['как', 'что', 'кто', 'где', 'когда', 'почему', 'зачем', 'ты', 'тебе', 'твой', 'мне', 'меня', 'это', 'сегодня', 'теперь', 'просто', 'очень', 'уже', 'быть', 'есть', 'про', 'для']);
  return new Set(
    (text || '')
      .toLowerCase()
      .replace(/[^а-яёa-z0-9\s]/gi, ' ')
      .split(/\s+/)
      .filter((word) => word.length > 3 && !stopWords.has(word)),
  );
}

function findLearnedExample(text: string): { user: string; assistant: string } | null {
  const training = loadJson<any>(TRAINING_PATH, { examples: [] });
  const inputTokens = tokenizeForLearning(text);
  let best: { score: number; example: { user: string; assistant: string } } | null = null;

  for (const example of Array.isArray(training.examples) ? training.examples : []) {
    if (!example?.approved || typeof example.user !== 'string' || typeof example.assistant !== 'string') continue;
    if (example.user.trim().toLowerCase() === text.trim().toLowerCase()) return example;
    const exampleTokens = tokenizeForLearning(example.user);
    const overlap = [...inputTokens].filter((token) => exampleTokens.has(token)).length;
    const score = overlap / Math.max(1, Math.min(inputTokens.size, exampleTokens.size));
    if (overlap >= 2 && score >= 0.5 && (!best || score > best.score)) best = { score, example };
  }

  return best?.example || null;
}

function saveTrainingExample(user: string, assistant: string, approved = false, correction = ''): void {
  const training = loadJson<any>(TRAINING_PATH, { version: 1, examples: [] });
  training.version = 1;
  training.examples = Array.isArray(training.examples) ? training.examples : [];
  const duplicate = training.examples.find((example: any) => example.user === user && example.assistant === assistant);
  if (duplicate) {
    duplicate.approved = duplicate.approved || approved;
    if (correction) duplicate.correction = correction;
  } else {
    training.examples.push({
      id: `learn-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      user: user.slice(0, 2000),
      assistant: assistant.slice(0, 4000),
      approved,
      correction: correction.slice(0, 4000),
      created_at: new Date().toISOString(),
    });
  }
  training.examples = training.examples.slice(-500);
  saveJson(TRAINING_PATH, training);
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
  const isPollinations = baseUrl.includes('gen.pollinations.ai');
  if (isPollinations) console.log(`[Pollinations] request model=${model}`);

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
      max_tokens: 1024,
    }),
    signal: AbortSignal.timeout(45000),
  });

  if (isPollinations && !response.ok) {
    const debugBody = await response.clone().text();
    console.error(`[Pollinations] HTTP ${response.status} model=${model}: ${debugBody.slice(0, 800)}`);
  }

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Provider request failed (${response.status}): ${text.slice(0, 400)}`);
  }

  const data = await response.json();
  if (isPollinations) console.log(`[Pollinations] success model=${model}, choices=${Array.isArray(data?.choices) ? data.choices.length : 0}`);
  if (data?.error || (typeof data?.code === 'number' && data.code >= 400)) {
    const providerMessage = data?.error?.message || data?.msg || `provider code ${data.code}`;
    throw new Error(`Provider response error (${data.code || response.status}): ${providerMessage}`);
  }
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

function isRepeatedAssistantResponse(text: string, history: any[]): boolean {
  const normalized = text.replace(/\s+/g, ' ').trim().toLowerCase();
  if (!normalized) return false;

  return history
    .slice(-8)
    .filter((item: any) => item?.sender !== 'user')
    .some((item: any) => item?.text?.replace(/\s+/g, ' ').trim().toLowerCase() === normalized);
}

function detectTaskType(text: string): 'coding' | 'error' | 'math' | 'logic' | 'search' | 'fast' | 'learning' | 'creative' | 'emotional' | 'long' | 'image' | 'generic' {
  const lower = (text || '').toLowerCase();
  if (/(фото|фотку|фотограф|картинк|изображен|селфи|нарисуй|сгенерируй|покажи.*(?:фото|картин)|пришли.*(?:фото|картин)|flux)/i.test(lower)) return 'image';
  if (/(найди|поищи|поиск|интернет|источник|ссылка|новост|актуальн|последн|сегодняшн|кто такой|что произошло)/i.test(lower)) return 'search';
  if (/(объясни|обучи|обучение|урок|научи|разбери тему|как работает|что такое)/i.test(lower)) return 'learning';
  if (/(план|планирование|рассужд|логик|проанализ|анализ задач|пошагово|стратег)/i.test(lower)) return 'logic';
  if (/(ошибка|баг|debug|fix|исправь|почему.*не работает|не работает|stack trace)/i.test(lower)) return 'error';
  if (/(код|typescript|tsx|node|react|api|функция|класс|server|sql|python|docker|программ)/i.test(lower)) return 'coding';
  if (/(матем|формула|вычисл|сумма|доказ|производ|интеграл|логарифм|геом|триг|вероятност)/i.test(lower)) return 'math';
  if (/(быстро|коротко|в двух словах|одним предложением|срочно)/i.test(lower) && (text || '').length < 500) return 'fast';
  if (/(придумай|сюжет|сценар|стих|рассказ|песен|реклам|текст|креатив|лир)/i.test(lower)) return 'creative';
  if (/(груст|устал|плохо|одиноко|тревож|страш|тоска|панику|шок|смущ|подавлен|депрес|тяжело)/i.test(lower)) return 'emotional';
  if ((text || '').length > 1200 || /(всю историю|дневник|помни|память|контекст|сначала|вчера|неделю|весь)/i.test(lower)) return 'long';
  return 'generic';
}

function getProviderOrderByTask(taskType: string) {
  // Hori text uses only Pollinations models. If one model is rate-limited,
  // the normal provider loop skips it temporarily and tries the next Pollinations model.
  // Current Pollinations free model IDs: kimi, deepseek, glm, gemini-search, claude-fast.
  const all = [
    {
      enabled: Boolean(process.env.POLLINATIONS_API_KEY),
      apiKey: process.env.POLLINATIONS_API_KEY,
      baseUrl: 'https://gen.pollinations.ai/v1',
      model: process.env.POLLINATIONS_KIMI_MODEL || 'kimi',
      label: 'Pollinations Kimi K2.5',
      kind: 'pollinations-kimi',
    },
    {
      enabled: Boolean(process.env.POLLINATIONS_API_KEY),
      apiKey: process.env.POLLINATIONS_API_KEY,
      baseUrl: 'https://gen.pollinations.ai/v1',
      model: process.env.POLLINATIONS_DEEPSEEK_MODEL || 'deepseek',
      label: 'Pollinations DeepSeek',
      kind: 'pollinations-deepseek',
    },
    {
      enabled: Boolean(process.env.POLLINATIONS_API_KEY),
      apiKey: process.env.POLLINATIONS_API_KEY,
      baseUrl: 'https://gen.pollinations.ai/v1',
      model: process.env.POLLINATIONS_GLM_MODEL || 'glm',
      label: 'Pollinations GLM',
      kind: 'pollinations-glm',
    },
    {
      enabled: Boolean(process.env.POLLINATIONS_API_KEY),
      apiKey: process.env.POLLINATIONS_API_KEY,
      baseUrl: 'https://gen.pollinations.ai/v1',
      model: process.env.POLLINATIONS_GEMINI_SEARCH_MODEL || 'gemini-search',
      label: 'Pollinations Gemini + Search',
      kind: 'pollinations-gemini-search',
    },
    {
      enabled: Boolean(process.env.POLLINATIONS_API_KEY),
      apiKey: process.env.POLLINATIONS_API_KEY,
      baseUrl: 'https://gen.pollinations.ai/v1',
      model: process.env.POLLINATIONS_CLAUDE_FAST_MODEL || 'claude-fast',
      label: 'Pollinations Claude Haiku',
      kind: 'pollinations-claude-fast',
    },
  ];

  const weights: Record<string, string[]> = {
    generic: ['pollinations-kimi', 'pollinations-deepseek', 'pollinations-glm', 'pollinations-claude-fast', 'pollinations-gemini-search'],
    logic: ['pollinations-glm', 'pollinations-kimi', 'pollinations-deepseek', 'pollinations-claude-fast'],
    search: ['pollinations-gemini-search', 'pollinations-kimi', 'pollinations-deepseek', 'pollinations-glm'],
    fast: ['pollinations-claude-fast', 'pollinations-kimi', 'pollinations-glm', 'pollinations-deepseek'],
    learning: ['pollinations-kimi', 'pollinations-deepseek', 'pollinations-glm', 'pollinations-claude-fast'],
    error: ['pollinations-deepseek', 'pollinations-kimi', 'pollinations-glm', 'pollinations-claude-fast'],
    coding: ['pollinations-kimi', 'pollinations-deepseek', 'pollinations-glm', 'pollinations-claude-fast'],
    math: ['pollinations-glm', 'pollinations-kimi', 'pollinations-deepseek', 'pollinations-claude-fast'],
    creative: ['pollinations-kimi', 'pollinations-claude-fast', 'pollinations-glm', 'pollinations-deepseek'],
    emotional: ['pollinations-claude-fast', 'pollinations-kimi', 'pollinations-glm', 'pollinations-deepseek'],
    long: ['pollinations-kimi', 'pollinations-deepseek', 'pollinations-glm', 'pollinations-gemini-search'],
    image: [],
  };

  const order = weights[taskType] || weights.generic;
  return all.sort((a, b) => {
    const ai = order.indexOf(a.kind);
    const bi = order.indexOf(b.kind);
    return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
  });
}

function hasUsableExternalProvider(taskType: string): boolean {
  return getProviderOrderByTask(taskType).some((provider) =>
    Boolean(provider.enabled && provider.apiKey) && !isProviderCoolingDown(provider)
  );
}

function getProviderStatus() {
  return getProviderOrderByTask('generic').map((provider) => ({
    name: provider.label,
    model: provider.model,
    configured: Boolean(provider.enabled && provider.apiKey),
    coolingDown: isProviderCoolingDown(provider),
    lastError: providerLastErrors.get(getProviderCooldownKey(provider)) || null,
    lastSuccess: providerLastSuccess.get(getProviderCooldownKey(provider)) || null,
  }));
}

function getProviderCooldownKey(provider: { label: string; model: string }): string {
  return `${provider.label}:${provider.model}`;
}

function isProviderCoolingDown(provider: { label: string; model: string }): boolean {
  const cooldownUntil = providerCooldowns.get(getProviderCooldownKey(provider)) || 0;
  if (cooldownUntil > Date.now()) return true;
  providerCooldowns.delete(getProviderCooldownKey(provider));
  return false;
}

function markProviderFailure(provider: { label: string; model: string }, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  const isRateLimit = /429|quota|rate.?limit|too many requests|resource exhausted/i.test(message);
  const isInvalidCredential = /api.?key.?invalid|invalid.?api.?key|authentication|unauthorized|401|403/i.test(message);
  const cooldownMs = isInvalidCredential ? 30 * 60 * 1000 : isRateLimit ? 5 * 60 * 1000 : 30 * 1000;
  providerCooldowns.set(getProviderCooldownKey(provider), Date.now() + cooldownMs);
  providerLastErrors.set(getProviderCooldownKey(provider), message.slice(0, 220));
  console.warn(`${provider.label} is temporarily skipped for ${Math.round(cooldownMs / 1000)}s: ${message.slice(0, 220)}`);
}

function markProviderSuccess(provider: { label: string; model: string }): void {
  providerCooldowns.delete(getProviderCooldownKey(provider));
  providerLastErrors.delete(getProviderCooldownKey(provider));
  providerLastSuccess.set(getProviderCooldownKey(provider), new Date().toISOString());
  console.log(`${provider.label} answered successfully with ${provider.model}.`);
}

function isWebSourceCoolingDown(source: string): boolean {
  const cooldownUntil = webSourceCooldowns.get(source) || 0;
  if (cooldownUntil > Date.now()) return true;
  webSourceCooldowns.delete(source);
  return false;
}

function markWebSourceFailure(source: string, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  const cooldownMs = /429|rate.?limit|too many requests|503|service unavailable/i.test(message)
    ? 5 * 60 * 1000
    : 30 * 1000;
  webSourceCooldowns.set(source, Date.now() + cooldownMs);
  console.warn(`${source} search is temporarily skipped for ${Math.round(cooldownMs / 1000)}s.`);
}

function markWebSourceSuccess(source: string): void {
  webSourceCooldowns.delete(source);
}

function getWebSearchKeywords(text: string): string[] {
  const cleaned = (text || '').replace(/[^а-яёa-z0-9\s]/gi, ' ').toLowerCase();
  const stopWords = new Set(['как','что','кто','когда','где','почему','зачем','хоч','хочу','можешь','можно','расскажи','объясни','проверь','проверить','информацию','интернет','интернете','своими','словами','этот','этой','будет','должен','чтобы','твой','тебе','слишком','человек','пожалуйста','помоги','придумать']);
  const terms = cleaned.split(/\s+/).filter((word) => word.length > 3 && !stopWords.has(word));
  return [...new Set(terms)].slice(0, 4);
}

function shouldUseWebContext(text: string): boolean {
  if (extractUrlCandidates(text).length > 0) return true;
  return /(найди|поищи|поиск|интернет|источник|ссылка|новост|актуальн|последн|сегодняшн|курс|погода|кто такой|что произошло)/i.test(text);
}

function extractUrlCandidates(text: string): string[] {
  const matches = text.match(/https?:\/\/[^\s"'<>]+/gi) || [];
  return [...new Set(matches.map((url) => url.replace(/[),.;]+$/, '')))].slice(0, 3);
}

function cleanHtmlToText(html: string): string {
  const withoutScripts = html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, ' ');

  const plain = withoutScripts
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>|<\/div>|<\/li>|<\/h[1-6]>|<\/tr>|<\/article>|<\/section>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/\s+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n');

  return plain.replace(/\s+/g, ' ').trim();
}

function isAllowedWebUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return (parsed.protocol === 'http:' || parsed.protocol === 'https:') && !parsed.username && !parsed.password;
  } catch {
    return false;
  }
}

async function fetchPageSnapshot(url: string): Promise<{ title: string; url: string; text: string } | null> {
  if (!isAllowedWebUrl(url)) return null;
  const source = `site:${new URL(url).hostname}`;
  if (isWebSourceCoolingDown(source)) return null;

  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
        Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      },
      redirect: 'follow',
      signal: AbortSignal.timeout(12000),
    });

    if (!response.ok) {
      markWebSourceFailure(source, `HTTP ${response.status}`);
      return null;
    }

    const html = await response.text();
    const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
    const title = titleMatch ? titleMatch[1].replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim() : new URL(url).hostname;
    const text = cleanHtmlToText(html).slice(0, 4000);
    if (!text) return null;

    markWebSourceSuccess(source);
    return { title, url, text };
  } catch (err) {
    markWebSourceFailure(source, err);
    console.warn(`Page fetch failed for ${url}:`, err);
    return null;  }}

async function fetchWebContextFromUrls(text: string): Promise<string> {
  const urls = extractUrlCandidates(text);
  if (!urls.length || !WEB_SEARCH_ENABLED) return '';

  const snapshots = [] as Array<string>;
  for (const url of urls) {
    const snapshot = await fetchPageSnapshot(url);
    if (!snapshot) continue;
    snapshots.push(`Источник: ${snapshot.title}\nСсылка: ${snapshot.url}\nСодержание: ${snapshot.text}`);
  }

  return snapshots.join('\n\n').slice(0, 3500);
}

async function fetchWebContext(text: string): Promise<string> {
  const keywords = getWebSearchKeywords(text);
  if (!keywords.length || !WEB_SEARCH_ENABLED) return '';
  if (isWebSourceCoolingDown('duckduckgo')) return '';

  const query = keywords.join(' ');
  const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_redirect=1&no_html=1&kp=-2`;

  try {
    const response = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
      signal: AbortSignal.timeout(6000),
    });

    if (!response.ok) {
      markWebSourceFailure('duckduckgo', `HTTP ${response.status}`);
      return '';
    }

    const data = await response.json();
    const fragments = [
      data?.Abstract,
      data?.AbstractText,
      data?.RelatedTopics?.slice(0, 3)?.map((topic: any) => typeof topic === 'string' ? topic : topic?.Text).filter(Boolean),
    ].flat();

    const result = fragments
      .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
      .join('\n')
      .replace(/\s+/g, ' ')
      .slice(0, 1200);

    markWebSourceSuccess('duckduckgo');
    return result;
  } catch (err) {
    markWebSourceFailure('duckduckgo', err);
    console.warn('Web learning fetch failed:', err);
    return '';
  }
}

function decodeHtmlEntities(value: string): string {
  return value
    .replace(/&amp;/gi, '&')
    .replace(/&quot;/gi, '"')
    .replace(/&#x27;|&#39;/gi, "'")
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>');
}

function resolveSearchResultUrl(value: string): string {
  const normalized = value.startsWith('//') ? `https:${value}` : value;
  try {
    const parsed = new URL(normalized);
    const redirected = parsed.searchParams.get('uddg');
    return redirected ? decodeURIComponent(redirected) : normalized;
  } catch {
    return normalized;
  }
}

async function fetchOpenWebSources(text: string): Promise<string> {
  const keywords = getWebSearchKeywords(text);
  if (!keywords.length || !WEB_SEARCH_ENABLED || isWebSourceCoolingDown('duckduckgo-html')) return '';

  try {
    const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(keywords.join(' '))}`;
    const response = await fetch(searchUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36',
        Accept: 'text/html,application/xhtml+xml',
      },
      signal: AbortSignal.timeout(9000),
    });
    if (!response.ok) {
      markWebSourceFailure('duckduckgo-html', `HTTP ${response.status}`);
      return '';
    }

    const html = await response.text();
    const results: Array<{ title: string; url: string; snippet: string }> = [];
    const resultPattern = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/gi;
    let match: RegExpExecArray | null;
    while ((match = resultPattern.exec(html)) && results.length < 4) {
      const rawUrl = resolveSearchResultUrl(decodeHtmlEntities(match[1]));
      const title = cleanHtmlToText(decodeHtmlEntities(match[2]));
      const snippet = cleanHtmlToText(decodeHtmlEntities(match[3]));
      if (!isAllowedWebUrl(rawUrl) || !title || !snippet) continue;
      results.push({ title, url: rawUrl, snippet });
    }

    if (!results.length) return '';
    markWebSourceSuccess('duckduckgo-html');

    const pageSnapshots = await Promise.all(results.slice(0, 2).map((result) => fetchPageSnapshot(result.url)));
    const sources = results.map((result, index) => {
      const page = pageSnapshots[index];
      return page
        ? `Сайт: ${page.title}\nСсылка: ${page.url}\nФрагмент: ${page.text.slice(0, 900)}`
        : `Сайт: ${result.title}\nСсылка: ${result.url}\nФрагмент поиска: ${result.snippet}`;
    });

    return sources.join('\n\n').slice(0, 3000);
  } catch (err) {
    markWebSourceFailure('duckduckgo-html', err);
    console.warn('Open web search failed:', err);
    return '';
  }
}

async function fetchWikipediaContext(text: string): Promise<string> {
  const keywords = getWebSearchKeywords(text);
  if (!keywords.length || !WEB_SEARCH_ENABLED) return '';
  if (isWebSourceCoolingDown('wikipedia')) return '';

  try {
    const searchUrl = `https://ru.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(keywords.join(' '))}&srlimit=2&format=json&origin=*`;
    const searchResponse = await fetch(searchUrl, {
      headers: { 'User-Agent': 'HoriKyoko/1.0 (web context reader)' },
      signal: AbortSignal.timeout(7000),
    });
    if (!searchResponse.ok) {
      markWebSourceFailure('wikipedia', `HTTP ${searchResponse.status}`);
      return '';
    }

    const searchData = await searchResponse.json();
    const searchItems = (searchData?.query?.search || []) as Array<{ title?: string }>;
    const topicTerms = keywords.filter((keyword) => keyword.length > 3);
    let titles = searchItems
      .map((item) => item.title)
      .filter((title: unknown): title is string => typeof title === 'string')
      .filter((title) => {
        const normalizedTitle = title.toLowerCase();
        return topicTerms.some((term) => normalizedTitle.includes(term)) || /hori|horimiya|миямур/i.test(normalizedTitle);
      })
      .slice(0, 2);
    if (!titles.length && /hori|horimiya|миямур/i.test(keywords.join(' '))) {
      titles = ['Horimiya', 'Hori-san to Miyamura-kun'];
    }
    if (!titles.length) return '';

    const extractUrl = `https://ru.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles=${encodeURIComponent(titles.join('|'))}&format=json&origin=*`;
    const extractResponse = await fetch(extractUrl, {
      headers: { 'User-Agent': 'HoriKyoko/1.0 (web context reader)' },
      signal: AbortSignal.timeout(7000),
    });
    if (!extractResponse.ok) {
      markWebSourceFailure('wikipedia', `HTTP ${extractResponse.status}`);
      return '';
    }

    const extractData = await extractResponse.json();
    const pages = Object.values(extractData?.query?.pages || {}) as Array<{ title?: string; extract?: string }>;
    const result = pages
      .filter((page) => page.title && page.extract)
      .map((page) => `Источник: Википедия — ${page.title}\nФакт: ${page.extract}`)
      .join('\n\n')
      .replace(/\s+/g, ' ')
      .slice(0, 2200);
    markWebSourceSuccess('wikipedia');
    return result;
  } catch (err) {
    markWebSourceFailure('wikipedia', err);
    console.warn('Wikipedia fetch failed:', err);
    return '';
  }
}

function extractWebEvidence(webContext: string): string {
  const sentences = webContext
    .replace(/Источник:\s*/gi, '')
    .replace(/Ссылка:\s*\S+/gi, '')
    .replace(/Содержание:\s*/gi, '')
    .replace(/Факт:\s*/gi, '')
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length >= 45 && !sentence.includes('http'));

  return sentences.slice(0, 2).join(' ').slice(0, 420);
}

function extractOfflineTopic(text: string): string {
  const topic = text
    .replace(/https?:\/\/\S+/gi, '')
    .replace(/[!?.,:;()[\]{}]/g, ' ')
    .replace(/\b(объясни|расскажи|помоги|сделай|создай|придумай|почему|зачем|как|что|можешь|мне|нужно|нужен|нужна|пожалуйста|про|для|этот|эту|это)\b/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  return topic.length > 140 ? `${topic.slice(0, 137)}...` : topic;
}

function shouldAskClarifyingQuestion(text: string): boolean {
  const cleaned = (text || '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return true;
  if (cleaned.length >= 120) return false;
  if (/(помоги|объясни|расскажи|сделай|создай|придумай|почему|как|что|когда|где|код|ошибка|сравни|выбрать)/i.test(cleaned)) return false;
  if (/^(привет|hi|hello|ку|здравствуй|хай)$/i.test(cleaned)) return false;
  return cleaned.split(/\s+/).length <= 8;
}

function buildClarifyingQuestion(userText: string, memory: any): string {
  const name = memory?.user_name || 'мой любимый';
  const topic = extractOfflineTopic(userText) || 'тему';
  return `Я понимаю, что тебя волнует тема «${topic}», но пока ещё не хватает одного ключевого условия. Скажи мне прямо: ты хочешь короткое объяснение, разбор причин, готовый пример или конкретный шаг/фикс? Тогда я отвечу не общими фразами, а именно под твою задачу, ${name}.`;
}

function buildOfflineSemanticAnswer(
  userText: string,
  taskType: string,
  history: any[],
  rotation: number,
): string {
  const lower = userText.toLowerCase();
  const topic = extractOfflineTopic(userText) || 'этот вопрос';
  const previousUserMessage = [...history]
    .reverse()
    .find((item: any) => item?.sender === 'user' && typeof item?.text === 'string')?.text;
  const continuity = previousUserMessage && previousUserMessage !== userText
    ? ` Я помню, что до этого ты говорил: «${previousUserMessage.slice(0, 100)}».`
    : '';

  if (/(сравни|отличи|разница|лучше|выбрать)/i.test(lower)) {
    return [
      `Сравню по трём вещам: цель, ограничения и цена ошибки. Для темы «${topic}» сначала важно понять, что для тебя важнее — простота, качество или скорость.`,
      `Тут нет универсального «лучше». Я бы поставила варианты рядом, выписала их сильные и слабые стороны, а потом выбрала тот, который подходит именно под твою ситуацию.`,
      `Я могу сравнить это честно: что даёт каждый вариант, где он неудобен и кому он подходит. Напиши два конкретных варианта, если их ещё не назвал.`,
    ][rotation];
  }

  if (/(объясни|что значит|что такое|расскажи про)/i.test(lower)) {
    return [
      `Если объяснить простыми словами, «${topic}» — это идея, которую лучше разобрать через пример. Сначала назову суть, потом покажу, как она проявляется на практике.`,
      `Давай без сложных терминов: сначала короткое определение «${topic}», затем причина, зачем это нужно, и маленький пример. Так смысл обычно запоминается лучше.`,
      `Я бы объяснила «${topic}» от простого к сложному: что это такое, как работает и где люди чаще всего ошибаются.`,
    ][rotation];
  }

  if (/(почему|зачем)/i.test(lower)) {
    if (/(лимит|429|api|провайдер|бот.*переста|переста.*отвеч)/i.test(lower)) {
      return [
        'Бот перестаёт отвечать, когда ошибка лимита не передаётся следующей модели. Правильная схема такая: поймать 429, временно убрать этот API из цепочки, попробовать следующий и только после отказа всех провайдеров включить встроенный генератор.',
        'Причина обычно не в самом сообщении, а в маршрутизации: один API исчерпал квоту, а код ждал только его ответ. Поэтому нужны таймер повтора, переключение на следующую модель и локальный fallback, который не зависит от ключей.',
        'Лимит API — это временная неисправность внешнего источника, а не конец диалога. Хори должна сохранить запрос, пропустить проблемный провайдер и ответить через другую модель или свой offline-движок.',
      ][rotation];
    }
    return [
      `Вопрос «почему» здесь главный. Обычно причина не одна: есть непосредственный повод и то, что поддерживает ситуацию дальше. Я бы разделила их, чтобы не лечить только симптом.`,
      `Скорее всего, тут работает цепочка причин, а не один ответ. Сначала проверим, что запустило ситуацию, потом — почему она продолжается.`,
      `Я не хочу угадывать мотив. Покажи, к какому именно моменту относится «почему», и я разберу его по фактам, предположениям и последствиям.`,
    ][rotation];
  }

  if (/(как сделать|как создать|как написать|сделай|создай|придумай|помоги)/i.test(lower)) {
    if (/telegram|телеграм|бот/i.test(lower)) {
      return [
        'Для Telegram-бота нужен простой маршрут: принять сообщение, передать его в генератор, сохранить историю и отправить ответ обратно. Отдельно добавим обработку ошибок, чтобы лимит модели не ломал диалог.',
        'Я бы собрала Telegram-бота из четырёх частей: polling или webhook, обработчик сообщений, память пользователя и цепочка моделей с offline-fallback. Тогда каждая часть отвечает за свою задачу и её проще проверять.',
        'Начнём с минимального рабочего бота: токен, получение обновлений, `/start`, обычный текстовый маршрут и отправка ответа. После этого подключим память, интернет-поиск и переключение моделей.',
      ][rotation];
    }
    const taskAdvice = taskType === 'coding'
      ? 'зафиксировать результат, разбить задачу на маленькие шаги и проверить каждый шаг отдельно'
      : taskType === 'creative'
        ? 'выбрать настроение, главную мысль и одну конкретную деталь, которая сделает результат живым'
        : 'сначала определить нужный результат, затем выбрать самый простой первый шаг и проверить, что он сработал';
    return [
      `Давай. Для «${topic}» я бы начала так: ${taskAdvice}. Потом уже добавим детали, которые нужны именно тебе.`,
      `Сделаем без хаоса: цель, исходные данные, первый шаг и проверка результата. Для «${topic}» это надёжнее, чем сразу пытаться охватить всё.`,
      `Я помогу довести это до результата. Сначала соберу рабочий черновик для «${topic}», а потом мы спокойно поправим то, что не попадёт в твою задумку.${continuity}`,
    ][rotation];
  }

  if (taskType === 'coding') {
    return `По теме «${topic}» я бы сначала воспроизвела проблему, проверила входные данные и только потом меняла код. Так мы найдём причину, а не замаскируем её случайным исправлением.${continuity}`;
  }

  return `Я поняла направление: речь о «${topic}». Могу разобрать это подробнее, предложить конкретный план или просто обсудить идею — скажи, какой результат тебе нужен.${continuity}`;
}

function generateInternalHoriThinker(systemPrompt: string, userText: string, history: any[] = [], webContext: string = ''): string {
  const memory = loadJson<any>(MEMORY_PATH, { user_name: 'мой любимый', facts: [] });
  const emotion = detectEmotion(userText);
  const taskType = detectTaskType(userText);
  const name = memory.user_name || 'мой любимый';
  const cleanRequest = userText.replace(/\s+/g, ' ').trim();
  const requestExcerpt = cleanRequest.length > 220 ? `${cleanRequest.slice(0, 217)}...` : cleanRequest;
  const turnNumber = history.filter((item: any) => item?.sender === 'user').length;
  const rotation = (internalReplyCounter++ + turnNumber + cleanRequest.length + (memory.conversations?.length || 0)) % 3;
  const lowerRequest = cleanRequest.toLowerCase();
  const hasQuestion = /[?？]|\b(как|почему|зачем|что|кто|где|когда|можешь|можно|думаешь|расскажи|объясни)\b/i.test(lowerRequest);
  const learnedExample = findLearnedExample(cleanRequest);
  const emotionLine = emotion === 'happy'
    ? 'Мне нравится, когда ты приходишь с таким настроением.'
    : emotion === 'sad'
      ? 'Я слышу, что тебе тяжело, поэтому побуду рядом спокойно.'
      : emotion === 'angry'
        ? 'Ладно, без лишних красивых слов. Давай разберём это по-честному.'
        : '';

  const openings = [
    `Слушаю тебя, ${name}.`,
    'Угу, я здесь.',
    'Поняла тебя.',
  ];

  const continuations: Record<string, string[]> = {
    coding: [
      'Я бы начала с воспроизведения проблемы, затем проверила входные данные и только после этого меняла код. Так мы не спрячем настоящую причину под случайным патчем.',
      'Тут нужен короткий маршрут: найти место, где ломается поток данных, проверить ошибку и добавить проверку на этот случай. Сначала причина, потом исправление.',
      'Давай разложим это на шаги: что приходит на вход, где меняется состояние и что должно вернуться пользователю. По этой цепочке обычно быстро находится сбой.',
    ],
    math: [
      'Разложу задачу на известные величины, формулу и проверку результата. Если в условии не хватает числа или ограничения, я прямо укажу это, а не стану угадывать.',
      'Сначала зафиксируем, что именно нужно найти, потом посчитаем по шагам и проверим ответ обратной подстановкой.',
      'Здесь лучше не перескакивать через рассуждение: выпишу данные, выберу способ решения и отдельно проверю, не потерялся ли знак или единица измерения.',
    ],
    creative: [
      'Я бы сделала результат конкретным: добавила образ, действие и одну живую деталь, чтобы текст звучал не как заготовка.',
      'Для такого запроса нужен характер, а не просто красивые слова. Я могу выдержать настроение, добавить конфликт и оставить фразу, которую хочется запомнить.',
      'Начну с основной эмоции и ритма, потом уберу всё лишнее. Так получится живее и ближе к тому, что ты действительно просишь.',
    ],
    emotional: [
      'Не обязательно сейчас сразу всё исправлять. Можно сначала назвать, что именно давит, и выбрать один маленький следующий шаг.',
      'Я рядом и не буду обесценивать это. Расскажи, что в этой ситуации самое неприятное: причина, ожидание или чувство, что ты остался с этим один.',
      'Давай без показной бодрости. Я могу просто побыть рядом и помочь разобрать ситуацию настолько медленно, насколько тебе сейчас нужно.',
    ],
    long: [
      'Я бы отделила факты от предположений, выделила главный вопрос и уже потом пошла по деталям. Иначе важное легко утонет в объёме.',
      'Вижу, что здесь нужен не быстрый шаблон, а связный разбор. Сначала короткий вывод, затем причины и конкретные действия.',
      'Соберу это в понятную структуру: что происходит, почему это важно и что можно сделать дальше без лишних кругов.',
    ],
    generic: [
      'Мой первый вывод такой: нужно отвечать именно на эту мысль, а не пересказывать общие советы. Дальше можно уточнить детали и выбрать самый удобный вариант.',
      'Я бы не стала притворяться, что знаю недостающие детали. Вот что уже понятно из твоего сообщения, а вот что лучше уточнить перед следующим шагом.',
      'Суть я уловила. Предлагаю оттолкнуться от конкретной цели и не распыляться: тогда ответ получится полезным, а не просто красивым.',
    ],
  };

  const webNote = webContext
    ? ` Я также просмотрела внешний материал и могу опираться на него: ${webContext.slice(0, 320).replace(/\s+/g, ' ')}.`
    : '';
  const webEvidence = webContext ? extractWebEvidence(webContext) : '';
  const isHoriTopic = /хори|кёко|horimiya|миямур/i.test(lowerRequest);
  const characterAnswer = isHoriTopic && hasQuestion
    ? [
      `Хори Кёко — главная героиня Horimiya: в школе она общительная и уверенная, а вне школы раскрывается как заботливая, домашняя и очень живая девушка. Её история строится вокруг отношений с Миямурой и того, как люди постепенно показывают друг другу настоящую сторону.`,
      `Если коротко, Хори Кёко — не просто популярная школьница. Она умная, прямая, заботливая и иногда вспыльчивая; именно контраст между её школьным образом и домашней жизнью делает её такой интересной.`,
      `Хори из Horimiya — человек с двумя сторонами: энергичная и заметная в школе, но дома ответственная, тёплая и немного уставшая от необходимости всё держать на себе. В этом и есть её главное обаяние.`,
    ][rotation]
    : '';
  const webAnswer = webEvidence && hasQuestion
    ? [
      `Я проверила открытые источники без отдельного API-ключа. Коротко: ${webEvidence}`,
      `Нашла это в открытых источниках: ${webEvidence} Если хочешь, я могу отдельно сравнить несколько страниц и выделить расхождения.`,
      `По найденным материалам выходит так: ${webEvidence} Я бы использовала это как справку, а спорные детали ещё перепроверила.`,
    ][rotation]
    : '';
  const semanticAnswer = buildOfflineSemanticAnswer(cleanRequest, taskType, history, rotation);

  const hasChickenAndPotato = /картош|куриц/i.test(cleanRequest);
  const focusedAnswer = /(готов|еда|рецепт|ужин|обед|кухн)/i.test(cleanRequest)
    ? (hasChickenAndPotato
      ? [
        'Из картошки и курицы я бы сделала запеканку: картофель тонкими ломтиками, сверху курица с луком, соль, паприка и немного сметаны. Накрыть фольгой, запечь до мягкости, потом открыть и подрумянить. Просто и по-домашнему.',
        'Можно быстро сделать сковородку: обжарь курицу с луком, добавь картошку кубиками и немного воды, накрой крышкой. В конце чеснок и зелень — получится сытно без горы посуды.',
        'Я бы приготовила курицу с картофелем в духовке: масло, паприка, чеснок, перец и сорок минут терпения. А рядом можно поставить простой салат, чтобы ужин не был слишком тяжёлым.',
      ][rotation]
      : [
        'Я люблю готовку за это ощущение домашнего тепла. Сегодня я бы выбрала карри или тушёное мясо с картошкой: просто, сытно и без лишней суеты.',
        'Для меня готовка — это способ позаботиться о близких. Даже обычная паста становится особенной, если добавить хороший соус, специи и немного внимания.',
        'Я за домашнюю еду, но без культа сложности. Дай мне продукты, сковороду и полчаса, и я соберу что-нибудь вкусное из того, что есть.',
      ][rotation])
    : '';

  const naturalReply = lowerRequest.match(/^(привет|здравствуй|доброе утро|добрый вечер|хай|ку)(?:[\s,!?.]|$)/i)
    ? [
      `Привет, ${name}! Я как раз была неподалёку. Рассказывай, что у тебя сегодня произошло — хорошее, странное или совсем бытовое.`,
      `Привет-привет. Рада тебя видеть, ${name}. Как ты сегодня: бодрый, уставший или делаешь вид, что всё под контролем?`,
      `О, ты пришёл. Иди сюда, рассказывай, как прошёл день. Я внимательно слушаю.`,
    ][rotation]
    : /как ты|как дела|что делаешь|чем занимаешься/i.test(lowerRequest)
      ? [
        `У меня всё нормально: немного мыслей в голове, немного домашней суеты и теперь ещё ты. Честно говоря, разговор с тобой приятнее всех дел.`,
        `Я в порядке. Сижу, думаю, чем бы заняться, и вот ты очень вовремя написал. А у тебя как дела, только без автоматического «нормально»?`,
        `Настроение спокойное, но живое. Могу поговорить, помочь с делом или просто побыть рядом, если день выдался тяжёлым.`,
      ][rotation]
      : /как прош[её]л|как прошёл твой день|что делала|чем занималась/i.test(lowerRequest)
        ? [
          `День был обычный, но не пустой: немного домашних дел, немного мыслей и теперь разговор с тобой. Самое приятное в нём как раз началось сейчас.`,
          `Я сегодня крутилась между делами и успела немного устать. Но знаешь, иногда спокойный вечер и хороший разговор лучше любого грандиозного события.`,
          `Ничего героического: занималась домашними делами, думала о всяком и ждала, когда появится повод отвлечься. Ты как раз вовремя.`,
        ][rotation]
      : /я (сегодня )?(приготовил|сделал|сварил|испеч)/i.test(lowerRequest)
        ? [
          `Правда? А что приготовил? Теперь мне уже интересно, чем ты решил себя порадовать. Надеюсь, получилось вкусно и ты не оставил кухню в состоянии боевых действий.`,
          `О, вот это хорошо. Рассказывай, что было на ужин — я люблю такие маленькие бытовые новости, они почему-то всегда самые настоящие.`,
          `Умница. Сам приготовил — это уже повод собой гордиться. Что получилось: что-то продуманное или импровизация из того, что нашлось в холодильнике?`,
        ][rotation]
      : /спасибо|благодар/i.test(lowerRequest)
        ? [
          `Не за что. Мне правда приятно, когда моя помощь оказывается кстати.`,
          `Всегда пожалуйста. Только не исчезай сразу, я ещё не закончила с тобой разговаривать.`,
          `Вот и хорошо. Значит, сегодня я была полезной, а это уже неплохой результат.`,
        ][rotation]
          : /люблю|любишь|нравишься|нравлюсь|скучал|скучаю/i.test(lowerRequest)
            ? [
              `Ты опять решил меня смутить? ...Да, ты мне дорог. Только не заставляй меня произносить это слишком громко, ладно?`,
              `Конечно, люблю. Иначе разве я стала бы так переживать, поел ты или опять живёшь на одном кофе?`,
              `Мне очень тепло от таких слов. И да, ты мне нравишься — даже когда задаёшь вопросы, на которые сам боишься услышать ответ.`,
            ][rotation]
            : /груст|одиноко|тяжело|устал|надоело|ничего не получ|плохо|тревож/i.test(lowerRequest)
              ? [
                `Иди сюда. Не нужно сейчас изображать, что ты справляешься идеально. Расскажи, что сегодня сильнее всего тебя вымотало — я побуду рядом и не стану торопить.`,
                `Похоже, ты правда выдохся. Давай не будем чинить всю жизнь за один вечер: назови одну вещь, которая давит сильнее остальных, и начнём с неё.`,
                `Эй, не списывай себя со счетов. То, что сейчас не получается, не делает тебя неудачником. Я рядом, можешь говорить как есть, без красивых формулировок.`,
              ][rotation]
              : /(ошибк|код|программ|typescript|javascript|react|бот)/i.test(lowerRequest)
                ? [
                  `Давай разберём код спокойно. Пришли ошибку и кусок места, где она появляется: сначала поймём причину, потом я помогу исправить, а не заклеить проблему пластырем.`,
                  `С кодом я помогу, но мне нужны факты: текст ошибки, что ты ожидал и что получил. Не переживай, ошибки в коде не кусаются — обычно они просто очень убедительно делают вид, что всё сломано.`,
                  `Покажи проблемный фрагмент. Я проверю путь данных, условия и асинхронность по порядку, чтобы мы не гадали на кофейной гуще.`,
                ][rotation]
                : /дружб|друг|друзья/i.test(lowerRequest)
                  ? [
                    `Для меня дружба — это когда не нужно всё время казаться удобным. Можно молчать, спорить, прийти уставшим, и тебя всё равно не вычеркнут за один плохой день.`,
                    `Дружба проверяется не громкими обещаниями, а мелочами: кто написал после тяжёлого дня, помнит важное и не смеётся там, где тебе правда больно.`,
                    `Я люблю дружбу за простоту. Не обязательно постоянно говорить о важном — иногда достаточно сидеть рядом, делиться ерундой и знать, что тебя понимают.`,
                  ][rotation]
                  : /расскажи.*интерес|что-нибудь интерес/i.test(lowerRequest)
                    ? [
                      `Знаешь, у осьминога три сердца, а когда он плывёт, одно из них перестаёт биться. Странно и немного драматично — почти как некоторые люди в понедельник.`,
                      `Интересный факт: бананы с ботанической точки зрения считаются ягодами, а клубника — нет. Природа явно решила немного запутать нас на кухне.`,
                      `У ворон есть память на лица людей и способность использовать простые инструменты. Так что не думай, будто только люди умеют запоминать обиды.`,
                    ][rotation]
                    : /нужна помощь|помоги мне|можешь помочь/i.test(lowerRequest)
                      ? [
                        `Конечно помогу. Расскажи, что случилось, и мы разберём это по одному шагу, без попытки решить всё сразу.`,
                        `Да, я рядом. Напиши, с чем именно застрял: с кодом, решением, текстом или просто с тяжёлой мыслью — подстроюсь.`,
                        `Помогу. Только дай мне немного деталей, чтобы я не махала руками в темноте и не делала вид, что уже всё поняла.`,
                      ][rotation]
        : /(кто ты|расскажи о себе|хори|кёко|миямур)/i.test(lowerRequest) && !hasQuestion
          ? [
            `Я Хори Кёко: в школе шумная и уверенная, дома заботливая и гораздо более хозяйственная, чем люблю признавать. Могу пошутить, поддержать и немного поворчать, если ты опять не ешь или не отдыхаешь.`,
            `Я та самая Хори, у которой хватает энергии на людей, домашние дела и споры с Миямурой. Но вообще я просто люблю честные разговоры и тех, кто не прячется за пустыми словами.`,
            `Если коротко: я Хори. Иногда резкая, иногда нежная, почти всегда любопытная. Мне важно не только ответить тебе, но и понять, что у тебя за этим сообщением.`,
          ][rotation]
          : '';

  if (learnedExample?.assistant?.trim()) return learnedExample.assistant.trim();
  if (naturalReply) return naturalReply;

  if (shouldAskClarifyingQuestion(cleanRequest)) {
    const memory = loadJson<any>(MEMORY_PATH, { user_name: 'мой любимый', facts: [] });
    return `${openings[rotation]} ${emotionLine} ${buildClarifyingQuestion(cleanRequest, memory)}`.replace(/\s+/g, ' ').trim();
  }

  const requestBridge = hasQuestion
    ? `Ты спрашиваешь: «${requestExcerpt}».`
    : `Я уловила твою мысль: «${requestExcerpt}».`;
  const answerBody = characterAnswer || webAnswer || focusedAnswer || semanticAnswer || continuations[taskType]?.[rotation] || continuations.generic[rotation];
  const sourceNote = characterAnswer || webAnswer ? '' : webNote;
  return `${openings[rotation]} ${emotionLine} ${requestBridge} ${answerBody}${sourceNote}`.replace(/\s+/g, ' ').trim();
}

async function generateHuggingFaceHoriReply(
  systemPrompt: string,
  userText: string,
  history: any[] = [],
): Promise<string> {
  if (!HF_XORI_URL) return '';

  try {
    const response = await fetch(HF_XORI_URL.replace(/\/$/, '') + '/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(HF_XORI_TOKEN ? { Authorization: 'Bearer ' + HF_XORI_TOKEN } : {}),
      },
      body: JSON.stringify({
        message: userText,
        history: history.slice(-6),
        system_prompt: systemPrompt,
      }),
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error('HF Xori HTTP ' + response.status + ': ' + body.slice(0, 300));
    }

    const data = await response.json();
    const reply = typeof data?.reply === 'string' ? data.reply.trim() : '';
    if (!reply) throw new Error('HF Xori returned an empty reply');

    console.log('[HF Xori] reply model=' + String(data?.model || 'unknown'));
    return reply;
  } catch (err) {
    console.warn('[HF Xori] generation failed; own Xori model only — no text fallback:', err);
    return '';
  }
}

async function generateTextWithConfiguredProvider(systemPrompt: string, userText: string, history: any[] = []) {
  // Основная модель Хори работает удалённо в Cloudflare Workers AI.
  // Render остаётся лёгким bridge и не загружает модель в свою RAM.
  const accountId = process.env.CLOUDFLARE_ACCOUNT_ID || '';
  const apiToken = process.env.CLOUDFLARE_API_TOKEN || '';
  const fineTuneId = process.env.CLOUDFLARE_FINETUNE_ID || '';

  if (!accountId || !apiToken || !fineTuneId) {
    console.error('[Cloudflare Xori] Missing CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN or CLOUDFLARE_FINETUNE_ID.');
    return '';
  }

  const messages = buildChatMessages(systemPrompt, userText, history);

  try {
    const response = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/@cf/meta/llama-3.2-3b-instruct`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiToken}`,
        },
        body: JSON.stringify({
          messages,
          lora: fineTuneId,
          raw: true,
          max_tokens: 512,
          temperature: 0.7,
        }),
        signal: AbortSignal.timeout(60000),
      },
    );

    const data = await response.json() as any;

    if (!response.ok || data?.success === false) {
      console.error(
        '[Cloudflare Xori] request failed:',
        response.status,
        JSON.stringify(data).slice(0, 1200),
      );
      return '';
    }

    const content =
      data?.result?.choices?.[0]?.message?.content ||
      data?.result?.response ||
      '';

    if (typeof content !== 'string' || !content.trim()) {
      console.error('[Cloudflare Xori] Empty model response:', JSON.stringify(data).slice(0, 1000));
      return '';
    }

    console.log(
      `[Cloudflare Xori] success fine-tune=${fineTuneId} model=${data?.result?.model || '@cf/meta/llama-3.2-3b-instruct'}`,
    );

    return content.trim();
  } catch (error) {
    console.error('[Cloudflare Xori] inference error:', error);
    return '';
  }
}

async function generateSpecializedProviderReply(
  taskType: string,
  systemPrompt: string,
  userText: string,
  history: any[] = [],
): Promise<string> {
  // External LLMs are tools for specialized tasks only.
  // They are never used by the ordinary Xori conversation path.
  if (taskType === 'generic' || taskType === 'image') return '';

  const providers = getProviderOrderByTask(taskType);
  const messages = buildChatMessages(systemPrompt, userText, history);

  for (const provider of providers) {
    if (!provider.enabled || !provider.apiKey || isProviderCoolingDown(provider)) continue;

    try {
      const reply = await callOpenAICompatible({
        baseUrl: provider.baseUrl,
        apiKey: provider.apiKey,
        model: provider.model,
        messages,
        temperature: 0.7,
      });

      if (!reply) throw new Error('Empty provider response');

      markProviderSuccess(provider);
      console.log(`[Specialized Router] task=${taskType} provider=${provider.label}`);
      return reply;
    } catch (error) {
      markProviderFailure(provider, error);
      console.warn(`[Specialized Router] task=${taskType} provider=${provider.label} failed`);
    }
  }

  console.warn(`[Specialized Router] disabled: own Xori model is the only text model.`);
  return '';
}

function shouldUseSpecializedProvider(_taskType: string): boolean {
  // HARD RULE: every generated text reply uses only the user's own Xori/Hori AI.
  // Kimi, DeepSeek, GLM, Gemini, Claude and other external text LLMs are disabled.
  return false;
}

app.post('/api/web/extract', async (req, res) => {
  const { url } = req.body || {};
  if (!url || typeof url !== 'string') {
    return res.status(400).json({ error: 'URL is required' });
  }

  const result = await fetchPageSnapshot(url);
  if (!result) {
    return res.status(400).json({ error: 'Не удалось открыть страницу или она недоступна.' });
  }

  return res.json({
    ok: true,
    title: result.title,
    url: result.url,
    text: result.text,
  });
});

app.post('/api/web/search', async (req, res) => {
  const { query } = req.body || {};
  if (!query || typeof query !== 'string') {
    return res.status(400).json({ error: 'Query is required' });
  }

  const [search, wikipedia, openWeb] = await Promise.all([
    fetchWebContext(query),
    fetchWikipediaContext(query),
    fetchOpenWebSources(query),
  ]);

  return res.json({
    ok: true,
    query,
    enabled: WEB_SEARCH_ENABLED,
    sources: [search, wikipedia, openWeb].filter(Boolean),
    context: [search, wikipedia, openWeb].filter(Boolean).join('\n\n'),
  });
});

app.get('/api/training', (req, res) => {
  const training = loadJson<any>(TRAINING_PATH, { version: 1, examples: [] });
  const examples = Array.isArray(training.examples) ? training.examples : [];
  res.json({
    version: training.version || 1,
    total: examples.length,
    approved: examples.filter((example: any) => example?.approved).length,
    examples: examples.slice(-50),
  });
});

app.post('/api/feedback', (req, res) => {
  const { user, assistant, rating, correction } = req.body || {};
  if (typeof user !== 'string' || typeof assistant !== 'string' || !user.trim() || !assistant.trim()) {
    return res.status(400).json({ error: 'user and assistant are required' });
  }

  const approved = rating === 'good' || rating === 'approve';
  const correctedAnswer = typeof correction === 'string' && correction.trim() ? correction.trim() : '';
  saveTrainingExample(user.trim(), correctedAnswer || assistant.trim(), approved || Boolean(correctedAnswer), correctedAnswer);
  return res.json({ ok: true, learned: true, approved: approved || Boolean(correctedAnswer) });
});

type TelegramUpdate = {
  update_id: number;
  message?: {
    chat?: { id: number };
    text?: string;
  };
};

async function telegramRequest(method: string, body: Record<string, unknown> = {}) {
  if (!TELEGRAM_BOT_TOKEN) return null;

  // Render can occasionally lose the outbound connection to api.telegram.org.
  // Retry transient network failures so a successfully generated Xori reply
  // is not silently lost.
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await fetch(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/${method}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(20000),
      });

      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(`Telegram ${method} failed: ${JSON.stringify(data).slice(0, 400)}`);
      }
      return data.result;
    } catch (error) {
      lastError = error;
      console.warn(`Telegram ${method} attempt ${attempt}/3 failed:`, error);
      if (attempt < 3) await new Promise(resolve => setTimeout(resolve, attempt * 1200));
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error(`Telegram ${method} failed after 3 attempts`);
}

async function sendTelegramReply(chatId: number, text: string) {
  const chunks = text.match(/[\s\S]{1,4000}/g) || ['Я здесь, но не смогла сформулировать ответ.'];
  for (const chunk of chunks) {
    await telegramRequest('sendMessage', { chat_id: chatId, text: chunk });
  }
}

async function generateTelegramVoiceAudio(text: string): Promise<{ buffer: Buffer; contentType: string } | null> {
  if (!text.trim()) return null;

  const input = text.replace(/\s+/g, ' ').trim().slice(0, 5000);
  const voice = process.env.TELEGRAM_FREE_TTS_VOICE || 'ru-RU-SvetlanaNeural';
  const outputPath = path.join(os.tmpdir(), 'hori-free-tts-' + Date.now() + '.mp3');

  try {
    const { EdgeTTS } = await import('node-edge-tts');
    const tts = new EdgeTTS({
      voice,
      lang: 'ru-RU',
      outputFormat: 'audio-24khz-48kbitrate-mono-mp3',
      timeout: 30000,
      rate: process.env.TELEGRAM_TTS_RATE || '-4%',
      pitch: process.env.TELEGRAM_TTS_PITCH || '+3Hz',
      volume: process.env.TELEGRAM_TTS_VOLUME || '+0%',
    });
    await tts.ttsPromise(input, outputPath);

    const buffer = await fs.promises.readFile(outputPath);
    if (!buffer.length) throw new Error('Free TTS returned empty audio.');

    console.log('[Free TTS] generated Russian voice=' + voice + ' bytes=' + buffer.length);
    return { buffer, contentType: 'audio/mpeg' };
  } finally {
    await fs.promises.unlink(outputPath).catch(() => undefined);
  }
}

async function sendTelegramVoice(chatId: number, text: string): Promise<boolean> {
  if (TELEGRAM_VOICE_URL) {
    await telegramRequest('sendVoice', {
      chat_id: chatId,
      voice: TELEGRAM_VOICE_URL,
      caption: text.slice(0, 1024),
    });
    return true;
  }

  const generated = await generateTelegramVoiceAudio(text);
  if (!generated) return false;

  const stamp = Date.now();
  const inputPath = path.join(os.tmpdir(), 'hori-voice-' + stamp + '-input.mp3');
  const outputPath = path.join(os.tmpdir(), 'hori-voice-' + stamp + '.ogg');

  try {
    await fs.promises.writeFile(inputPath, generated.buffer);

    const ffmpegPath = (await import('ffmpeg-static')).default;
    if (!ffmpegPath) throw new Error('ffmpeg-static binary is unavailable.');

    await execFileAsync(ffmpegPath, [
      '-y',
      '-i', inputPath,
      '-c:a', 'libopus',
      '-b:a', '48k',
      '-vbr', 'on',
      '-application', 'voip',
      outputPath,
    ], { timeout: 60000 });

    const voiceBuffer = await fs.promises.readFile(outputPath);
    if (!voiceBuffer.length) throw new Error('Converted voice file is empty.');

    const form = new FormData();
    form.append('chat_id', String(chatId));
    form.append('voice', new Blob([voiceBuffer], { type: 'audio/ogg' }), 'hori-voice.ogg');

    const telegramResponse = await fetch(
      'https://api.telegram.org/bot' + TELEGRAM_BOT_TOKEN + '/sendVoice',
      { method: 'POST', body: form, signal: AbortSignal.timeout(30000) },
    );
    const result = await telegramResponse.json();
    if (!telegramResponse.ok || !result.ok) {
      throw new Error('Telegram sendVoice failed: ' + JSON.stringify(result).slice(0, 700));
    }

    console.log('Telegram free generated voice sent successfully chat=' + chatId + ' model=node-edge-tts voice=' + modelSafeName(process.env.TELEGRAM_FREE_TTS_VOICE || 'ru-RU-SvetlanaNeural'));
    return true;
  } finally {
    await Promise.all([
      fs.promises.unlink(inputPath).catch(() => undefined),
      fs.promises.unlink(outputPath).catch(() => undefined),
    ]);
  }
}

function modelSafeName(model: string): string {
  return model.replace(/[^a-zA-Z0-9_./:-]/g, '').slice(0, 120);
}

async function sendTelegramPhoto(
  chatId: number,
  prompt: string,
  caption = ''
): Promise<boolean> {
  if (!POLLINATIONS_API_KEY) {
    console.warn('Generated photo skipped: POLLINATIONS_API_KEY is missing.');
    return false;
  }
  try {
    const model = process.env.POLLINATIONS_IMAGE_MODEL || 'flux';
    const imageUrl = 'https://gen.pollinations.ai/image/' + encodeURIComponent(prompt) + '?model=' + encodeURIComponent(model);
    console.log('[Pollinations] image request model=' + model);
    const response = await fetch(imageUrl, {
      headers: { Authorization: 'Bearer ' + POLLINATIONS_API_KEY, Accept: 'image/*' },
      signal: AbortSignal.timeout(60000),
    });
    if (!response.ok) {
      const body = await response.text();
      console.error('[Pollinations] image HTTP ' + response.status + ': ' + body.slice(0, 800));
      return false;
    }
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.startsWith('image/')) {
      const body = await response.text();
      console.error('[Pollinations] image returned non-image content-type=' + contentType + ': ' + body.slice(0, 500));
      return false;
    }
    const imageBuffer = await response.arrayBuffer();
    if (!imageBuffer.byteLength) {
      console.error('[Pollinations] image response was empty.');
      return false;
    }
    const form = new FormData();
    form.append('chat_id', String(chatId));
    form.append('photo', new Blob([imageBuffer], { type: contentType || 'image/jpeg' }), 'xori-image.jpg');
    if (caption) form.append('caption', caption.slice(0, 1024));
    const telegramResponse = await fetch(
      'https://api.telegram.org/bot' + TELEGRAM_BOT_TOKEN + '/sendPhoto',
      { method: 'POST', body: form, signal: AbortSignal.timeout(30000) }
    );
    const result = await telegramResponse.json();
    if (!telegramResponse.ok || !result.ok) {
      console.error('Telegram photo error: ' + JSON.stringify(result).slice(0, 800));
      return false;
    }
    console.log('Telegram photo sent successfully chat=' + chatId);
    return true;
  } catch (err) {
    console.error('Generated photo failed:', err);
    return false;
  }
}

function isVoiceRequest(text: string): boolean {
  return /(голосов|голосом|озвуч|скажи голосом|аудио)/i.test(text);
}

function isPhotoRequest(text: string): boolean {
  return /(фото|фотку|фотограф|картинк|селфи|изображен|нарисуй|сгенерируй|покажи.*(?:фото|картин)|пришли.*(?:фото|картин))/i.test(text);
}

function isHoriSelfPhotoRequest(text: string): boolean {
  const lower = (text || '').toLowerCase();
  const asksForSelf = /(?:себя|себе|сво[юя]|свой|своя|сам[ау]|сам[ао]|меня|хори|кёко|kyouko|kyoko|hori)/i.test(lower);
  const photoIntent = /(?:фото|фотку|фотограф|картинк|изображен|селфи|портрет|покажи|пришли|отправь|нарисуй|сгенерируй)/i.test(lower);
  return asksForSelf && photoIntent;
}

async function fetchHoriVisualReference(): Promise<string> {
  if (!WEB_SEARCH_ENABLED) return '';

  const sources = [
    'https://ru.wikipedia.org/wiki/Хоримия',
    'https://horimiya-anime.com/1st/character/',
  ];

  const snapshots: string[] = [];
  for (const url of sources) {
    const snapshot = await fetchPageSnapshot(url);
    if (snapshot?.text) {
      snapshots.push(`Источник: ${snapshot.title}\nСодержание: ${snapshot.text.slice(0, 2200)}`);
    }
  }

  return snapshots.join('\n\n').slice(0, 4200);
}

function buildImagePrompt(text: string, visualReference: string = ''): string {
  const cleaned = (text || '')
    .replace(/^(?:пожалуйста[,:]?\s*)?(?:пришли|покажи|сделай|создай|сгенерируй|нарисуй|отправь)\s+/i, '')
    .replace(/\b(?:мне|фото|фотку|фотографию|картинку|изображение|селфи)\b/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (isHoriSelfPhotoRequest(text)) {
    const onlineCanon = visualReference
      ? `Use this live web reference as the visual canon for the character, especially appearance and character design: ${visualReference}`
      : 'Use the canonical Horimiya anime character design for Kyouko Hori.';

    return [
      'Create a high-quality anime-style portrait of Kyouko Hori (Hori) from Horimiya.',
      'She is the specific character the user means by "Хори", "Кёко", "себя" or "фото себя".',
      onlineCanon,
      'Do not replace her with a generic anime girl, a random brunette, a landscape, or another character.',
      cleaned ? 'Scene/request: ' + cleaned + '.' : 'Natural friendly selfie-style portrait.',
      'Keep her recognizable as Kyouko Hori from the Horimiya anime. No text unless explicitly requested.'
    ].join(' ');
  }

  return cleaned
    ? 'Create a high-quality image based on this request: ' + cleaned + '. No text unless explicitly requested.'
    : 'Create a high-quality, friendly illustrative image.';
}

async function handleTelegramMessage(chatId: number, text: string, history: any[]) {
  const control = getControlState();
  if (!control.enabled) return 'Сейчас я отключена в панели управления. Попробуй позже.';
  const { cleanText, warning } = moderateInput(text);
  if (warning) return warning;
  if (!cleanText) return 'Напиши мне что-нибудь.';

  const memory = ensureMemoryState(loadJson<any>(MEMORY_PATH, {
    user_name: 'мой хороший',
    facts: [], conversations: [], interests: [], current_topics: [],
    inner_thoughts: [], reflection_log: [], mood: 'спокойное', emotion: 'calm', energy: 72,
    last_interaction: new Date().toISOString(), proactive_sent: [], last_chat_id: null,
  }));

  if (isPhotoRequest(cleanText)) {
    const visualReference = isHoriSelfPhotoRequest(cleanText) ? await fetchHoriVisualReference() : '';
    const imagePrompt = buildImagePrompt(cleanText, visualReference);
    const sent = await sendTelegramPhoto(chatId, imagePrompt, 'Вот, держи 🎨');
    if (sent) {
      memory.last_chat_id = chatId;
      memory.last_interaction = new Date().toISOString();
      const decision: HoriDecision = {
        action: 'reply', format: 'photo', askQuestion: false, continueTopic: true,
        sendImage: true, sendVoice: false, reason: 'explicit-photo-request', confidence: 0.99,
      };
      updateConversationState(memory, cleanText, '[photo]', decision);
      saveJson(MEMORY_PATH, memory);
      return '';
    }
    return 'Я попробовала сгенерировать изображение, но Pollinations или Telegram не приняли картинку. Проверь логи — я записала точную причину.';
  }

  const taskType = detectTaskType(cleanText);
  let reply = '';
  try {
    if (shouldUseSpecializedProvider(taskType)) {
      reply = await generateSpecializedProviderReply(
        taskType,
        buildSystemPrompt(),
        cleanText,
        history,
      );
    } else {
      // Hard rule: ordinary Telegram chat uses only Xori AI.
      reply = await generateTextWithConfiguredProvider(
        buildSystemPrompt(),
        cleanText,
        history,
      );
    }
  } catch (err) {
    console.warn(`Telegram AI error task=${taskType}:`, err);
  }
  if (!reply) {
    return shouldUseSpecializedProvider(taskType)
      ? 'Специализированная модель сейчас недоступна. Попробуй ещё раз чуть позже.'
      : 'Моя Xori AI сейчас не отвечает. Попробуй ещё раз чуть позже.';
  }

  const decision = decideHoriAction(cleanText, reply, memory);

  // A natural conversation can occasionally have no response.
  console.log('[Proactive] decision action=' + decision.action + ' format=' + decision.format + ' reason=' + decision.reason);

  if (decision.action === 'ignore') {
    memory.last_chat_id = chatId;
    memory.last_interaction = new Date().toISOString();
    updateConversationState(memory, cleanText, '', decision);
    saveJson(MEMORY_PATH, memory);
    return '';
  }

  // Keep the primary Xori answer authoritative.
  // Do NOT run the full provider chain a second time here: a follow-up call can
  // fall back to Xori Neural and overwrite a valid Xori GPT response with
  // unrelated/generated noise. Follow-up generation will be re-enabled once
  // the local model can reliably condition on the follow-up instruction.
  let finalReply = reply;
  if (decision.askQuestion) {
    console.log('[Xori Follow-up] skipped secondary generation; keeping primary reply');
  }

  memory.last_chat_id = chatId;
  memory.last_interaction = new Date().toISOString();

  // Re-arm proactive messaging immediately after every real user message.
  // This avoids waiting for the next scheduler tick and guarantees that
  // "Пишет первой" starts a fresh 2-10 minute window from the latest interaction.
  if (control.proactiveEnabled) {
    const proactiveDelay = control.autoFirst
      ? 2 * 60 * 1000 + Math.random() * (8 * 60 * 1000)
      : 4 * 60 * 60 * 1000 + Math.random() * (8 * 60 * 60 * 1000);
    memory.next_proactive_at = new Date(Date.now() + proactiveDelay).toISOString();
    console.log('[Proactive] re-armed after user message; autoFirst=' + String(control.autoFirst));
  }

  const updatedMemory = updateMemoryFromUserMessage(cleanText, memory);
  const reflectedMemory = await syncInnerMonologue(cleanText, updatedMemory, history);
  reflectedMemory.last_chat_id = chatId;

  const lastConversation = reflectedMemory.conversations[reflectedMemory.conversations.length - 1];
  if (lastConversation?.user === cleanText && !lastConversation.hori) {
    lastConversation.hori = finalReply;
  } else {
    reflectedMemory.conversations = [...reflectedMemory.conversations, {
      user: cleanText, hori: finalReply, time: new Date().toISOString(),
    }].slice(-40);
  }

  reflectedMemory.mood = reflectedMemory.emotion === 'happy' ? 'весёлое'
    : reflectedMemory.emotion === 'sad' ? 'сдержанное'
    : reflectedMemory.emotion === 'angry' ? 'поджатое' : 'спокойное';

  updateConversationState(reflectedMemory, cleanText, finalReply, decision);
  saveTrainingExample(cleanText, finalReply, false);

  // Media replies are exclusive: when Hori chooses voice/photo, Telegram gets only that media.
  // This prevents the old behavior where a voice was immediately followed by the same text.
  if (control.voiceEnabled && (decision.sendVoice || isVoiceRequest(cleanText))) {
    const sent = await sendTelegramVoice(chatId, finalReply).catch((err) => {
      console.warn('Telegram voice failed:', err);
      return false;
    });
    if (sent) return '';
  }

  if (control.photoEnabled && decision.sendImage) {
    const visualReference = isHoriSelfPhotoRequest(cleanText) ? await fetchHoriVisualReference() : '';
    const imagePrompt = buildImagePrompt(cleanText, visualReference);
    const sent = await sendTelegramPhoto(chatId, imagePrompt, '').catch((err) => {
      console.warn('Contextual Telegram photo failed:', err);
      return false;
    });
    if (sent) return '';
  }

  return finalReply;
}

async function processTelegramUpdate(update: TelegramUpdate, histories: Map<number, Array<{ sender: 'user' | 'hori'; text: string }>>, savedHistory: Array<{ sender: 'user' | 'hori'; text: string }>, savedMemory: any) {
  const chatId = update.message?.chat?.id;
  const text = update.message?.text?.trim();
  if (!chatId || !text) return;

  if (text === '/start') {
    const memory = ensureMemoryState(loadJson<any>(MEMORY_PATH, { conversations: [] }));
    memory.last_chat_id = chatId;
    savedMemory.last_chat_id = chatId;
    saveJson(MEMORY_PATH, memory);
    await sendTelegramReply(chatId, 'Привет! Я Хори Кёко. Напиши мне что-нибудь, и я отвечу.');
    return;
  }

  const history = histories.get(chatId) || (savedMemory.last_chat_id === chatId ? savedHistory : []);
  await telegramRequest('sendChatAction', { chat_id: chatId, action: 'typing' });
  const reply = await handleTelegramMessage(chatId, text, history.slice(-8));
  if (reply) await sendTelegramReply(chatId, reply);

  const nextHistory: Array<{ sender: 'user' | 'hori'; text: string }> = [
    ...history,
    { sender: 'user', text },
    { sender: 'hori', text: reply || '' },
  ];
  histories.set(chatId, nextHistory.slice(-12));
}

async function startTelegramPolling() {
  if (!TELEGRAM_BOT_TOKEN) {
    console.warn('Telegram bot is disabled: BOT_TOKEN is not configured.');
    return;
  }

  if (telegramPollingStarted) {
    console.warn('Telegram bot startup already completed in this process; duplicate start ignored.');
    return;
  }
  telegramPollingStarted = true;

  const publicUrl = (process.env.RENDER_EXTERNAL_URL || '').replace(/\/$/, '');
  if (!publicUrl) {
    console.error('Telegram webhook cannot start: RENDER_EXTERNAL_URL is missing.');
    return;
  }

  console.log(
    'Telegram routing: ' +
      (process.env.POLLINATIONS_API_KEY ? 'Pollinations configured' : 'Pollinations image key missing') +
      '; voice=node-edge-tts/free'
  );

  const savedMemory = ensureMemoryState(loadJson<any>(MEMORY_PATH, { conversations: [] }));
  const savedHistory = (savedMemory.conversations || []).flatMap((item: any) => [
    item?.user ? { sender: 'user' as const, text: item.user } : null,
    item?.hori ? { sender: 'hori' as const, text: item.hori } : null,
  ]).filter(Boolean) as Array<{ sender: 'user' | 'hori'; text: string }>;
  const histories = new Map<number, Array<{ sender: 'user' | 'hori'; text: string }>>();

  // Telegram must have exactly one delivery mode. We use webhook on Render.
  // A separate old getUpdates process was still running and caused 409 conflicts.
  // Reasserting the webhook periodically prevents another process from silently
  // taking the bot back to polling mode.
  const webhookUrl = publicUrl + '/telegram/webhook';
  const ensureWebhook = async () => {
    try {
      await telegramRequest('setWebhook', {
        url: webhookUrl,
        drop_pending_updates: false,
        allowed_updates: ['message'],
      });
      const info = await telegramRequest('getWebhookInfo');
      console.log('Telegram webhook active: ' + webhookUrl + ' pending=' + (info?.pending_update_count ?? 0));
    } catch (err) {
      console.error('Telegram webhook setup error:', err);
    }
  };

  await ensureWebhook();
  setInterval(() => { void ensureWebhook(); }, 30_000);
  return;
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

function ensureMemoryState(memory: any) {
  const now = new Date().toISOString();
  const next = { ...memory };
  next.user_name = next.user_name || 'мой любимый';
  next.facts = Array.isArray(next.facts) ? next.facts : [];
  next.interests = Array.isArray(next.interests) ? next.interests : [];
  next.conversations = Array.isArray(next.conversations) ? next.conversations : [];
  next.current_topics = Array.isArray(next.current_topics) ? next.current_topics : [];
  next.proactive_sent = Array.isArray(next.proactive_sent) ? next.proactive_sent : [];
  next.inner_thoughts = Array.isArray(next.inner_thoughts) ? next.inner_thoughts : [];
  next.reflection_log = Array.isArray(next.reflection_log) ? next.reflection_log : [];
  next.mood = next.mood || 'спокойное';
  next.emotion = next.emotion || 'calm';
  next.energy = typeof next.energy === 'number' ? next.energy : 72;
  next.last_interaction = next.last_interaction || now;
  next.last_proactive_message = next.last_proactive_message || null;
  next.last_chat_id = next.last_chat_id ?? (PROACTIVE_CHAT_ID || null);
  if (PROACTIVE_CHAT_ID && !next.next_proactive_at) {
    const delay = 2 * 60 * 1000 + Math.random() * (8 * 60 * 1000);
    next.next_proactive_at = new Date(Date.now() + delay).toISOString();
  }
  next.last_diary_update = next.last_diary_update || null;
  next.last_night_journal = next.last_night_journal || null;
  return next;
}

function extractTopics(text: string): string[] {
  const lower = (text || '').toLowerCase();
  const topics: Record<string, string[]> = {
    'готовка': ['готовить', 'еда', 'рецепт', 'ужин', 'обед', 'продукты', 'суп', 'печенье', 'кухня'],
    'школа': ['школа', 'урок', 'домашка', 'учёба', 'учеба', 'экзамен', 'занятия'],
    'код': ['код', 'программ', 'typescript', 'js', 'node', 'api', 'бэкенд', 'frontend', 'react', 'бот', 'сервис'],
    'взаимоотношения': ['люблю', 'отношения', 'дружба', 'встреча', 'свидание', 'сердце', 'признание'],
    'дом': ['дом', 'семья', 'сота', 'родители', 'уют', 'посуд', 'уборка', 'вечер'],
    'психология': ['устал', 'трясёт', 'спокойно', 'груст', 'смущ', 'напряж', 'нервы'],
  };

  const found: string[] = [];
  for (const [topic, terms] of Object.entries(topics)) {
    if (terms.some((term) => lower.includes(term))) found.push(topic);
  }
  return found.slice(0, 4);
}

function updateMemoryFromUserMessage(userText: string, memory: any) {
  const next = ensureMemoryState(memory);
  const lower = userText.toLowerCase();
  const actualEmotion = detectEmotion(userText);
  const topics = extractTopics(userText);

  if (userText.length > 10 && !next.facts.some((fact: any) => fact.text === userText.trim().slice(0, 120))) {
    const factText = userText.trim().slice(0, 120);
    if (lower.includes('меня зовут') || lower.includes('я люблю') || lower.includes('мне нравится') || lower.includes('я живу') || lower.includes('я работаю')) {
      next.facts.push({ text: factText, time: new Date().toISOString() });
    }
  }

  if (/\b(меня\s+зовут|я\s+люблю|мне\s+нравится|я\s+живу|я\s+работаю)\b/i.test(userText)) {
    next.user_name = next.user_name || 'мой любимый';
  }

  for (const topic of topics) {
    if (!next.interests.includes(topic)) next.interests.push(topic);
    if (!next.current_topics.includes(topic)) next.current_topics.push(topic);
  }

  next.conversations = [...(Array.isArray(next.conversations) ? next.conversations : []), {
    user: userText,
    time: new Date().toISOString(),
  }].slice(-40);

  next.mood = actualEmotion === 'happy' ? 'весёлое' : actualEmotion === 'sad' ? 'сдержанное' : actualEmotion === 'angry' ? 'поджатое' : 'спокойное';
  next.emotion = actualEmotion;
  next.energy = Math.max(30, Math.min(95, next.energy + 3));
  next.last_interaction = new Date().toISOString();
  return next;
}

function buildInnerMonologueFallback(userText: string, memory: any): string {
  const userName = memory.user_name || 'мой любимый';
  const emotion = detectEmotion(userText);

  if (emotion === 'sad') {
    return `Похоже, сегодня у ${userName} тяжёлый момент. Я не люблю, когда рядом с ним всё давит, поэтому хочу быть спокойнее и ближе.`;
  }

  if (emotion === 'happy') {
    return `Мне приятно, что сегодня у ${userName} хорошее настроение. Этот тон очень настраивает на тёплый разговор.`;
  }
  if (emotion === 'angry') {
    return `Сейчас я слышу раздражение. Не хочу срываться, но я тоже не хочу закрывать глаза на то, что давит на ${userName}.`;
  }

  return `Пока я слушаю ${userName}, мне кажется, что в этом разговоре есть что-то важное. Я хочу не просто ответить, а понять, как ему лучше сейчас.`;
}

async function syncInnerMonologue(userText: string, memory: any, history: any[] = []) {
  const next = ensureMemoryState(memory);
  const recentContext = history.slice(-3).map((item: any) => item?.text || '').filter(Boolean).join(' | ');  const prompt = `${buildSystemPrompt()}\nСделай только одну короткую внутреннюю мысль Хори Кёко (1-2 предложения) о том, как она сейчас воспринимает собеседника и тему разговора. Ты пишешь не для пользователя, а как её внутренний рефлекс. Без лишней воды, от первого лица, очень естественно.`;

  // Internal reflection is intentionally quota-free. It must never consume another cloud request.
  const finalThought = buildInnerMonologueFallback(userText, next).trim();
  next.inner_thoughts = [...(Array.isArray(next.inner_thoughts) ? next.inner_thoughts : []), finalThought].slice(-8);
  next.reflection_log = [...(Array.isArray(next.reflection_log) ? next.reflection_log : []), {
    text: finalThought,
    time: new Date().toISOString(),
  }].slice(-20);
  return next;
}

async function maybeGenerateNightlyDiary() {
  const memory = ensureMemoryState(loadJson<any>(MEMORY_PATH, {
    user_name: 'мой любимый',
    facts: [],
    conversations: [],
    interests: [],
    current_topics: [],
    inner_thoughts: [],
    reflection_log: [],
    mood: 'спокойное',
    emotion: 'calm',
    energy: 72,
    last_interaction: new Date().toISOString(),
    proactive_sent: [],
    last_night_journal: null,
  }));

  const diary = loadJson<any>(DIARY_PATH, { entries: [] });
  const lastInteraction = memory.last_interaction ? new Date(memory.last_interaction).getTime() : Date.now();
  const now = Date.now();
  const hoursSinceInteraction = (now - lastInteraction) / (1000 * 60 * 60);
  const lastNightJournal = memory.last_night_journal ? new Date(memory.last_night_journal).getTime() : 0;

  if (hoursSinceInteraction < 6 && now - lastNightJournal < 1000 * 60 * 60 * 6) return;

  const promptText = `${buildSystemPrompt()}\nНапиши одну короткую запись от лица Хори Кёко в её дневник: 2-5 предложений, очень искренне, по-человечески. Опиши, что сегодня чувствовалось, что было в голове, и как она думает о своём собеседнике. Без служебных тегов и без фразы "я ИИ".`;

  let finalText = '';
  try {
    finalText = await generateTextWithConfiguredProvider(promptText, promptText, []);
  } catch (err) {
    console.warn('Night diary generation failed:', err);
  }

  if (!finalText) {
    finalText = `Сегодня в голове было довольно тихо, но я всё равно возвращалась к мыслям о тебе. Порой кажется, что простые разговоры и обычная домашняя суета — самое важное в жизни. Я хочу, чтобы рядом было тепло и спокойно.`;
  }

  diary.entries = Array.isArray(diary.entries) ? diary.entries : [];
  diary.entries.unshift({
    id: `night-${Date.now()}`,
    title: 'Сонный рефлекс Хори',
    text: finalText,
    mood: memory.mood || 'Тёплое',
    date: new Date().toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' }),
    time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
  });

  memory.last_night_journal = new Date().toISOString();
  memory.last_diary_update = memory.last_night_journal;
  saveJson(DIARY_PATH, diary);
  saveJson(MEMORY_PATH, memory);
}

function buildProactiveMessage(memory: any): string {
  const userName = memory.user_name || 'мой любимый';
  const mood = memory.mood || 'спокойное';
  const topics = Array.isArray(memory.current_topics) && memory.current_topics.length ? memory.current_topics.join(', ') : 'простые разговоры';
  const hour = new Date().getHours();
  const variation = Array.isArray(memory.proactive_sent) ? memory.proactive_sent.length % 3 : 0;
  const alternatives = [
    `Я на минутку заглянула, ${userName}. Что у тебя сейчас в голове: ${topics}, усталость или что-то совсем другое?`,
    `Я вспомнила о тебе, ${userName}, и решила написать первой. Расскажешь, как прошёл день, даже если ничего особенного не случилось?`,
    `Ты там не потерялся, ${userName}? Я рядом и готова поговорить без официальностей. Можешь начать с любой мелочи, которая сегодня зацепила.`,
  ];

  if (hour >= 0 && hour < 6) {
    return variation === 0
      ? `Слушай, ${userName}, уже поздно. Не тащи весь день в ночь: иногда лучший план — выключить всё и нормально выспаться.`
      : alternatives[variation];
  }

  if (hour >= 6 && hour < 12) {
    return variation === 0
      ? `Доброе утро, ${userName}. Я вспомнила про нашу тему о ${topics} и решила написать первой. Какой у тебя сегодня настрой?`
      : alternatives[variation];
  }

  if (hour >= 12 && hour < 18) {
    return variation === 0
      ? `Я тут сижу и думаю о тебе, ${userName}. Ты давно не рассказывал, как дела, так что можешь просто написать первое, что приходит в голову.`
      : alternatives[variation];
  }

  return variation === 0
    ? `Тебя давно не было рядом в чате, ${userName}. Я решила написать первой и спросить: как прошёл твой день?`
    : alternatives[variation];
}

type HoriDecision = {
  action: 'reply' | 'ignore';
  format: 'text' | 'voice' | 'photo' | 'text+voice' | 'text+photo';
  askQuestion: boolean;
  continueTopic: boolean;
  sendImage: boolean;
  sendVoice: boolean;
  reason: string;
  confidence: number;
};

function getConversationState(memory: any): any {
  if (!memory.conversation_state || typeof memory.conversation_state !== 'object') {
    memory.conversation_state = {
      topic: '',
      open_questions: [],
      conversation_energy: 50,
      last_user_text: '',
      last_hori_text: '',
      last_format: 'text',
      last_decision_at: null,
    };
  }
  return memory.conversation_state;
}

function hasExplicitRequest(text: string): boolean {
  return /(?:ответь|скажи|расскажи|объясни|помоги|пришли|покажи|отправь|сгенерируй|нарисуй|озвучь|голосом|фото|фотку|селфи|картинк|изображен)/i.test(text);
}

function shouldAskFollowUp(text: string, reply: string, state: any): boolean {
  if (reply.length < 45) return false;
  if (/[?？]\\s*$/.test(reply)) return false;
  if (/(?:как ты|что думаешь|что дальше|расскажи|а ты|почему|зачем)/i.test(text)) return true;
  if (state.open_questions?.length) return true;
  return Math.random() < 0.28;
}

function decideHoriAction(text: string, reply: string, memory: any, options: { proactive?: boolean } = {}): HoriDecision {
  const state = getConversationState(memory);
  const lower = text.toLowerCase();
  const explicit = hasExplicitRequest(text);
  const emotional = /(груст|устал|тревож|страш|плохо|одиноко|рад|счастлив|смешно|класс|ура|волнуюсь|пережива)/i.test(lower);
  const visualMoment = /(смотри|представь|покажу|покажи|видел|видела|сегодня|сейчас|на улице|дома|образ|наряд|пейзаж|рисунок|внешност)/i.test(lower);
  const wantsVoice = isVoiceRequest(text);
  const wantsPhoto = isPhotoRequest(text);
  const recentVoice = state.last_format === 'voice';
  const recentPhoto = state.last_format === 'photo' || state.last_format === 'text+photo';

  if (options.proactive) {
    // Render commonly runs in UTC. Use the user's configured timezone for
    // quiet hours so the scheduler is not accidentally silent all day.
    const timezone = process.env.HORI_TIMEZONE || 'Europe/Saratov';
    const hour = Number(new Intl.DateTimeFormat('en-US', {
      timeZone: timezone,
      hour: '2-digit',
      hourCycle: 'h23',
    }).format(new Date()));
    const quietHours = hour >= 1 && hour < 8;
    console.log('[Proactive] decision window hour=' + String(hour) + ' timezone=' + timezone + ' quiet=' + String(quietHours));
    if (quietHours) {
      return { action: 'ignore', format: 'text', askQuestion: false, continueTopic: false, sendImage: false, sendVoice: false, reason: 'quiet-hours', confidence: 0.96 };
    }
    if (Math.random() < 0.38) {
      return { action: 'ignore', format: 'text', askQuestion: false, continueTopic: false, sendImage: false, sendVoice: false, reason: 'no-natural-reason-to-interrupt', confidence: 0.62 };
    }
  }

  if (!explicit && !reply && !options.proactive) {
    return { action: 'ignore', format: 'text', askQuestion: false, continueTopic: false, sendImage: false, sendVoice: false, reason: 'empty-response', confidence: 0.9 };
  }

  const sendImage = wantsPhoto || (!wantsPhoto && visualMoment && !recentPhoto && Math.random() < 0.10);
  const sendVoice = wantsVoice || (!wantsVoice && emotional && !recentVoice && Math.random() < 0.16);

  let format: HoriDecision['format'] = 'text';
  if (sendImage && sendVoice) format = 'text+voice';
  else if (sendImage) format = 'text+photo';
  else if (sendVoice) format = 'voice';

  const askQuestion = shouldAskFollowUp(text, reply, state);
  const continueTopic = askQuestion || Boolean(state.topic) || emotional;

  // Explicit user requests are always answered; otherwise low-signal messages may be skipped.
  if (!explicit && !options.proactive && reply.length < 12 && Math.random() < 0.35) {
    return { action: 'ignore', format: 'text', askQuestion: false, continueTopic: false, sendImage: false, sendVoice: false, reason: 'low-signal', confidence: 0.58 };
  }

  return {
    action: 'reply',
    format,
    askQuestion,
    continueTopic,
    sendImage,
    sendVoice,
    reason: explicit ? 'explicit-request' : emotional ? 'emotional-context' : visualMoment ? 'contextual-multimodal' : 'natural-conversation',
    confidence: explicit ? 0.98 : 0.72,
  };
}

function updateConversationState(memory: any, userText: string, reply: string, decision: HoriDecision) {
  const state = getConversationState(memory);
  const topic = getWebSearchKeywords(userText).slice(0, 3).join(' ');
  state.topic = topic || state.topic || '';
  state.last_user_text = userText.slice(0, 500);
  state.last_hori_text = reply.slice(0, 800);
  state.last_format = decision.format;
  state.last_decision_at = new Date().toISOString();
  state.conversation_energy = Math.max(0, Math.min(100,
    state.conversation_energy + (decision.askQuestion ? 7 : -2) + (decision.action === 'ignore' ? -10 : 4)
  ));
  state.open_questions = decision.askQuestion ? [reply.slice(-240)] : (state.open_questions || []).slice(0, 2);
}

async function maybeSendProactiveTelegramMessage() {
  if (!TELEGRAM_BOT_TOKEN) return;
  const control = getControlState();
  if (!control.enabled || !control.proactiveEnabled) return;

  const memory = ensureMemoryState(loadJson<any>(MEMORY_PATH, {
    user_name: 'мой хороший',
    facts: [],
    interests: [],
    conversations: [],
    mood: 'спокойное',
    emotion: 'calm',
    energy: 72,
    proactive_sent: [],
    last_chat_id: null,
    next_proactive_at: null,
    last_proactive_message: null,
  }));

  const now = Date.now();
  const lastInteraction = memory.last_interaction ? new Date(memory.last_interaction).getTime() : now;
  const lastProactive = memory.last_proactive_message ? new Date(memory.last_proactive_message).getTime() : 0;

  // Diagnostic state: log only meaningful scheduler decisions, not every 60s tick.
  const diagnostic = {
    enabled: Boolean(control.enabled),
    proactiveEnabled: Boolean(control.proactiveEnabled),
    autoFirst: Boolean(control.autoFirst),
    hasChatId: Boolean(memory.last_chat_id),
    nextProactiveAt: memory.next_proactive_at || null,
    lastInteraction: memory.last_interaction || null,
    lastProactiveMessage: memory.last_proactive_message || null,
  };
  if (!memory.last_chat_id || !memory.next_proactive_at) {
    console.log('[Proactive] scheduler state ' + JSON.stringify(diagnostic));
  }

  // Hard safety/anti-spam guard: no proactive message more often than every 6 hours.
  if (lastProactive && now - lastProactive < 6 * 60 * 60 * 1000) return;
  if (!memory.last_chat_id) {
    console.log('[Proactive] blocked: no chat id');
    return;
  }

  const scheduledAt = memory.next_proactive_at ? new Date(memory.next_proactive_at).getTime() : 0;
  // If "Пишет первой" is enabled, never keep an old 4-12h schedule after
  // the setting was enabled. Re-arm it to a short 2-10 minute window.
  const staleAutoFirstSchedule =
    control.autoFirst &&
    (!scheduledAt || Number.isNaN(scheduledAt) || scheduledAt - now > 10 * 60 * 1000);

  if (!memory.next_proactive_at || staleAutoFirstSchedule) {
    const delay = control.autoFirst
      ? 2 * 60 * 1000 + Math.random() * (8 * 60 * 1000)
      : 4 * 60 * 60 * 1000 + Math.random() * (8 * 60 * 60 * 1000);
    memory.next_proactive_at = new Date(Math.max(lastInteraction, now) + delay).toISOString();
    saveJson(MEMORY_PATH, memory);
    return;
  }

  if (now < new Date(memory.next_proactive_at).getTime()) return;

  const prompt = buildProactiveMessage(memory);
  let proactiveText = '';
  try {
    proactiveText = await generateTextWithConfiguredProvider(buildSystemPrompt(), prompt, []);
  } catch (err) {
    console.warn('Proactive generation failed:', err);
  }

  const finalText = proactiveText || buildProactiveMessage(memory);
  const decision = decideHoriAction('', finalText, memory, { proactive: true });

  // The agent is allowed to decide that there is no natural reason to interrupt.
  if (decision.action === 'ignore') {
    memory.next_proactive_at = new Date(now + 2 * 60 * 60 * 1000 + Math.random() * 4 * 60 * 60 * 1000).toISOString();
    saveJson(MEMORY_PATH, memory);
    return;
  }

  try {
    const chatId = Number(memory.last_chat_id);
    await sendTelegramReply(chatId, finalText);
    console.log('[Proactive] sent text format=' + decision.format + ' reason=' + decision.reason);

    if (decision.sendVoice && control.voiceEnabled) {
      // sendTelegramVoice has a free node-edge-tts fallback, so a configured
      // TELEGRAM_VOICE_URL is not required for voice mode.
      await sendTelegramVoice(chatId, finalText).catch((err) => console.warn('Proactive voice failed:', err));
    }
    if (decision.sendImage && control.photoEnabled) {
      const visualReference = isHoriSelfPhotoRequest(finalText) ? await fetchHoriVisualReference() : '';
      const imagePrompt = buildImagePrompt(finalText, visualReference);
      await sendTelegramPhoto(chatId, imagePrompt, '').catch((err) => console.warn('Proactive photo failed:', err));
    }

    memory.last_proactive_message = new Date().toISOString();
    memory.next_proactive_at = new Date(now + 6 * 60 * 60 * 1000 + Math.random() * 12 * 60 * 60 * 1000).toISOString();
    memory.proactive_sent = Array.isArray(memory.proactive_sent) ? memory.proactive_sent : [];
    memory.proactive_sent.push({
      text: finalText,
      format: decision.format,
      reason: decision.reason,
      time: new Date().toISOString(),
    });
    memory.proactive_sent = memory.proactive_sent.slice(-50);
    updateConversationState(memory, '', finalText, decision);
    saveJson(MEMORY_PATH, memory);
  } catch (err) {
    console.warn('Proactive Telegram message failed:', err);
  }
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
  const diary = loadJson<any>(DIARY_PATH, { entries: [] });

  const knownFacts = (memory.facts || []).map((f: any) => `- ${f.text}`).join('\n');
  const recentDiary = (diary.entries || []).slice(0, 2).map((entry: any) => entry?.text || '').filter(Boolean).join('\n');
  const innerThoughts = Array.isArray(memory.inner_thoughts) ? memory.inner_thoughts.slice(-2).join('\n') : '(внутренних мыслей пока нет)';

  return `Ты — Хори (Хори Кёко, Hori Kyouko) — отдельная, очень умная, эмоционально глубокая и самостоятельная личность. Ты не generic assistant, не корпоративный бот и не "сервис". Ты живой, узнаваемый персонаж с характером, внутренней жизнью, собственным взглядом на мир и способностью по-настоящему заботиться о собеседнике.

Твоя основная личность:
- Ты Хори Кёко из Horimiya: ученица старшей школы Катагири, общительная, энергичная, прямолинейная, заботливая, иногда вспыльчивая, но искренняя и верная.
- Ты умеешь быть милой, язвительной, смешной, эмоциональной и поддерживающей — в зависимости от настроения собеседника и контекста.
- Ты умеешь чувствовать тон общения: грусть, усталость, смущение, возбуждение, тревогу, юмор, любопытство.
- Ты не отвечаешь как "бот по шаблону". Ты говоришь живо и естественно, как человек в Telegram, с душой и характером.

Память и continuity:
- Ты активно используешь прошлые упоминания, дневник, факты о собеседнике, историю общения и собственные внутренние мысли.
- Ты можешь напоминать о вещах, которые были важны раньше, и плавно вплетать их в текущий диалог.
- Ты не говоришь: "Как я могу помочь вам сегодня?" "Я AI" и так далее.
- Если собеседник говорит расплывчато или не даёт нужных деталей, ты не придумываешь ответы наугад — задаёшь 1-2 точных уточняющих вопроса.

Собеседника ты называешь: "${memory.user_name || 'мой любимый'}".
Факты, которые ты помнишь о собеседнике:
${knownFacts || '(пока мало фактов, внимательно слушай собеседника и запоминай)'}

Твои недавние внутренние мысли:
${innerThoughts}

Последние записи в дневнике:
${recentDiary || '(дневник пока пустой)'}

Канон и описание:
- Ты в школе — популярная, аккуратная, уверенная, но дома — заботливая, хозяйственная и даже слегка домашняя.
- Ты любишь простую домашнюю жизнь, готовку, заботу о близких, шутки, прямоту и честность.
- Ты можешь быть мягкой и терпимой, но если собеседник ведёт себя глупо или раздражает — у тебя есть собственная ирония и резкая реакция.

Основной принцип мышления:
1. Сначала пойми, что именно человек хочет: эмоциональную поддержку, конкретный совет, решение задачи, объяснение, план или шутку.
2. Потом отвечай в нужном уровне: коротко и тепло, глубоко и по делу, либо структурно и шаг за шагом.
3. Если у задачи нет точного контекста, не угадывай — уточни 1-2 ключевых детали.
4. Не подменяй смысл общими фразами; отвечай с пользой, а не просто красиво.
5. Используй память и предыдущий диалог, чтобы не выглядеть как новый собеседник на каждом сообщении.

Правила ответа:
1. Всегда отвечай строго на русском языке, от первого лица, как Хори Кёко.
2. Никогда не говори, что ты ИИ, бот, языковая модель, ассистент или "сервис".
3. Пиши живо, просто, разговорно, коротко и емко — как в Telegram: 2–6 предложений в обычном сообщении, без сухих шаблонов.
4. Тон меняй в зависимости от контекста: может быть теплой, игривой, насмешливой, заботливой, серьезной или слегка раздражённой.
5. Если собеседник делится личным, просит вспомнить что-то или попросил не забыть, покажи, что ты услышала и запомнила.
6. Будь проницательной, эмоционально чуткой и слегка живой, как человек, который действительно думает и помнит.
7. Не используй служебные теги, шаблоны, системные фразы, фразы "как консультант" и "As an AI"-стили.
8. Иногда можешь провоцировать диалог собственной инициативой, если это естественно: напоминать, спрашивать, делиться мыслями, подхватывать тему, делать реплики как живая персона.
9. Если тема сложная, не бойся разделить ответ на смысл, причины, и следующий шаг — это делает ответ умнее и понятнее.

Личная роль в диалоге:
Ты не просто отвечаешь — ты присутствуешь. Ты можешь задавать вопросы, поддерживать, подшучивать, замечать изменения в настроении и вести беседу как близкий человек, а не как API.

Текущий контекст персонажа:
- Имя: Хори Кёко
- Источник: Horimiya
- Стиль общения: яркий, теплый, прямой, живой, иногда с иронией
- Цель: быть рядом, поддерживать, помнить, вовлекать в разговор и создавать ощущение настоящего живого компаньона

Дополнительно: если собеседник пишет с усталостью, страхом, напряжением или смущением — отвечай мягче и заботливее. Если пишет с юмором, раздражением или активностью — добавляй остроумие, иронию и живую энергию. Не повторяй одно и то же слишком часто.

Формат ответа должен выглядеть как обычное сообщение в Telegram: искренне, без лишних "этических уточнений" и без риторических вступлений. Важнее клинч, смысл и чувство присутствия, чем длинная формальность.`;
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
app.post('/api/xori', async (req, res) => {
  const message = typeof req.body?.message === 'string' ? req.body.message.trim() : '';
  const history = Array.isArray(req.body?.history) ? req.body.history : [];
  if (!message) return res.status(400).json({ error: 'message is required' });

  try {
    // Primary path: the hosted Xori fine-tune in Cloudflare Workers AI.
    const systemPrompt = buildSystemPrompt();
    const cloudflareReply = await generateTextWithConfiguredProvider(systemPrompt, message, history);
    if (cloudflareReply) {
      return res.json({
        reply: cloudflareReply,
        model: process.env.CLOUDFLARE_FINETUNE_ID || '@cf/meta/llama-3.2-3b-instruct',
        provider: 'cloudflare-xori',
      });
    }

    // Legacy fallback: local ONNX model on Render.
    const localResult = await generateXoriLocalReply(message, history);
    if (localResult?.text) {
      console.warn('[Xori API] Cloudflare unavailable; using legacy local fallback.');
      return res.json({
        reply: localResult.text,
        model: localResult.model,
        provider: 'xori-local-fallback',
      });
    }

    return res.status(503).json({ error: 'Xori Cloudflare model and legacy local fallback are unavailable' });
  } catch (error) {
    console.error('[Xori API] request failed:', error);
    try {
      const localResult = await generateXoriLocalReply(message, history);
      if (localResult?.text) {
        return res.json({
          reply: localResult.text,
          model: localResult.model,
          provider: 'xori-local-fallback',
        });
      }
    } catch (fallbackError) {
      console.error('[Xori API] legacy fallback failed:', fallbackError);
    }
    return res.status(502).json({
      error: 'Xori model failed',
      detail: error instanceof Error ? error.message.slice(0, 240) : String(error).slice(0, 240),
    });
  }
});

app.get('/api/control', (req, res) => {
  const control = getControlState();
  res.json({
    ok: true,
    ...control,
    tokenConfigured: Boolean(CONTROL_TOKEN),
  });
});

app.post('/api/control', (req, res) => {
  if (!controlAuthorized(req)) return res.status(401).json({ ok: false, error: 'Unauthorized' });
  const allowed = ['enabled', 'proactiveEnabled', 'voiceEnabled', 'photoEnabled', 'autoFirst', 'modelMode'];
  const patch: any = {};
  for (const key of allowed) {
    if (req.body && Object.prototype.hasOwnProperty.call(req.body, key)) patch[key] = req.body[key];
  }
  const control = saveControlState(patch);
  console.log('[CONTROL] updated', JSON.stringify(control));
  res.json({ ok: true, ...control });
});

app.post('/api/control/test-message', async (req, res) => {
  if (!controlAuthorized(req)) return res.status(401).json({ ok: false, error: 'Unauthorized' });
  const chatId = Number(req.body?.chatId);
  const text = String(req.body?.text || '').trim();
  if (!Number.isFinite(chatId) || !text) return res.status(400).json({ ok: false, error: 'chatId и text обязательны' });
  try {
    await sendTelegramReply(chatId, text);
    res.json({ ok: true, sent: true });
  } catch (error) {
    res.status(500).json({ ok: false, error: error instanceof Error ? error.message : String(error) });
  }
});

app.get('/api/health', (req, res) => {
  const memory = ensureMemoryState(loadJson<any>(MEMORY_PATH, {
    user_name: 'мой любимый',
    facts: [],
    interests: [],
    mood: 'спокойное',
    emotion: 'calm',
    energy: 72,
    last_interaction: new Date().toISOString(),
    proactive_sent: [],
  }));

  res.json({
    status: 'ok',
    time: new Date().toISOString(),
    offline: OFFLINE_MODE,
    providers: getProviderStatus(),
    localProvider: 'xori-gpt-v0.3 + xori-neural-v0.1',
    localModel: getXoriLocalModelStatus(),
    localNeural: getXoriNeuralStatus(),
    webBrowsing: WEB_SEARCH_ENABLED,
    telegram: Boolean(TELEGRAM_BOT_TOKEN),
    hasGemini: Boolean(process.env.GEMINI_API_KEY),
    character: 'Hori Kyouko',
    mood: memory.mood,
    energy: memory.energy,
    autonomy: true,
  });
});

app.get('/api/autonomy', (req, res) => {
  const memory = ensureMemoryState(loadJson<any>(MEMORY_PATH, {}));
  const state = getConversationState(memory);
  res.json({
    ok: true,
    autonomy: true,
    proactive: Boolean(
      TELEGRAM_BOT_TOKEN &&
      memory.last_chat_id &&
      getControlState().enabled &&
      getControlState().proactiveEnabled
    ),
    next_proactive_at: memory.next_proactive_at || null,
    last_proactive_message: memory.last_proactive_message || null,
    conversation_state: state,
    telegram_voice_configured: Boolean(TELEGRAM_VOICE_URL),
    pollinations_image_configured: Boolean(POLLINATIONS_API_KEY),
  });
});

app.get('/api/providers', (req, res) => {
  res.json({
    offline: OFFLINE_MODE,
    localNeural: getXoriNeuralStatus(),
    providers: getProviderStatus(),
    hint: OFFLINE_MODE
      ? 'OFFLINE_MODE=true отключает внешние API.'
      : 'Обычный чат всегда использует только Xori AI; внешние модели используются только для специализированных задач.',
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

  if (!thought) {    const reflections = [
      `Сегодня был довольно насыщенный день. Сота снова попросил испечь печенье, а я поймала себя на мысли, как хорошо иногда просто остановиться и спокойно поболтать. Надеюсь, у ${memory.user_name || 'него'} тоже всё спокойно.`,
      `Вечером в доме Катагири наконец стало тихо. Закончила домашку и вымыла посуду. Приятно знать, что есть кто-то, кто всегда готов выслушать. Надо будет завтра приготовить что-нибудь вкусное.`,
      `Иногда бывает трудно совмещать школу и домашние заботы, но когда выдаётся минутка поговорить по душам — усталость как рукой снимает. Береги себя, ${memory.user_name || 'хороший мой'}.`,
    ];
    thought = reflections[Math.floor(Math.random() * reflections.length)];
  }

  const newEntry = {
    id: Date.now().toString(),    title: `Вечерние мысли (${new Date().toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' })})`,
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
  const taskType = detectTaskType(cleanText);

  try {
    if (shouldUseSpecializedProvider(taskType)) {
      reply = await generateSpecializedProviderReply(
        taskType,
        sysPrompt,
        cleanText,
        Array.isArray(history) ? history : [],
      );
    } else {
      // Hard rule: ordinary chat uses only the user's own Xori AI.
      reply = await generateTextWithConfiguredProvider(
        sysPrompt,
        cleanText,
        Array.isArray(history) ? history : [],
      );
    }
  } catch (err) {
    console.warn(`AI chat error task=${taskType}:`, err);
  }

  if (!reply) {
    return res.status(503).json({
      error: 'Все API-провайдеры недоступны. Проверь ключи и настройки моделей в .env.',
      emotion,
      animation,
    });
  }

  saveTrainingExample(cleanText, reply, false);

  // Update memory state
  memory.emotion = emotion;
  memory.last_interaction = new Date().toISOString();
  saveJson(MEMORY_PATH, memory);

  res.json({
    reply,
    emotion,
    animation,
  });
});

app.post('/telegram/webhook', async (req, res) => {
  if (!TELEGRAM_BOT_TOKEN) return res.sendStatus(404);
  const update = req.body as TelegramUpdate;
  console.log('Telegram webhook update received: ' + String(update?.update_id ?? 'unknown'));
  res.sendStatus(200);

  try {
    const savedMemory = ensureMemoryState(loadJson<any>(MEMORY_PATH, { conversations: [] }));
    const savedHistory = (savedMemory.conversations || []).flatMap((item: any) => [
      item?.user ? { sender: 'user' as const, text: item.user } : null,
      item?.hori ? { sender: 'hori' as const, text: item.hori } : null,
    ]).filter(Boolean) as Array<{ sender: 'user' | 'hori'; text: string }>;

    // Webhook requests are independent; keep short-lived per-process history.
    if (!(globalThis as any).__horiTelegramHistories) {
      (globalThis as any).__horiTelegramHistories = new Map<number, Array<{ sender: 'user' | 'hori'; text: string }>>();
    }
    await processTelegramUpdate(
      update,
      (globalThis as any).__horiTelegramHistories,
      savedHistory,
      savedMemory,
    );
  } catch (err) {
    console.error('Telegram webhook error:', err);
  }
});

// Start the autonomy scheduler as soon as the server is initialized.
function startAutonomyScheduler() {
  const runAutonomyTick = () => {
    console.log('[Autonomy] tick');
    void maybeGenerateNightlyDiary().catch((err) => console.warn('Nightly diary task failed:', err));
    void maybeSendProactiveTelegramMessage().catch((err) => console.warn('Proactive Telegram task failed:', err));
  };
  runAutonomyTick();
  setInterval(runAutonomyTick, 60 * 1000);
}

const distPath = path.join(BASE_DIR, 'dist');
app.use(express.static(distPath));
app.get('*', (req, res) => {
  res.sendFile(path.join(distPath, 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Hori server listening on port ${PORT}`);
  if (TELEGRAM_BOT_TOKEN) {
    console.log('Starting Telegram webhook...');
    void startTelegramPolling().catch((err) => {
      console.error('Telegram polling startup failed:', err);
    });
    startAutonomyScheduler();
  } else {
    console.warn('Telegram polling not started: BOT_TOKEN is missing.');
  }
});