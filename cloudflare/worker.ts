export interface Env {
  TELEGRAM_BOT_TOKEN: string;
  TELEGRAM_WEBHOOK_SECRET?: string;
  OPENAI_API_KEY?: string;
  OPENAI_BASE_URL?: string;
  OPENAI_MODEL?: string;
  ALLOWED_ORIGIN?: string;
}

const json = (data: unknown, status = 200, headers: HeadersInit = {}) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', ...headers },
  });

function cors(env: Env): HeadersInit {
  return {
    'Access-Control-Allow-Origin': env.ALLOWED_ORIGIN || '*',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  };
}

async function telegram(env: Env, method: string, body: unknown) {
  const response = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/${method}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`Telegram API ${response.status}: ${await response.text()}`);
  return response.json();
}

async function generateReply(env: Env, userText: string, history: Array<{ role: string; content: string }> = []) {
  if (!env.OPENAI_API_KEY) return 'AI backend пока не настроен. Добавьте OPENAI_API_KEY в секреты Cloudflare.';

  const baseUrl = (env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/$/, '');
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: env.OPENAI_MODEL || 'gpt-4o-mini',
      messages: [
        { role: 'system', content: 'Ты Xori — полезный русскоязычный AI-помощник. Отвечай кратко и по делу.' },
        ...history.slice(-8),
        { role: 'user', content: userText },
      ],
      temperature: 0.7,
      max_tokens: 1200,
    }),
  });
  if (!response.ok) throw new Error(`AI provider ${response.status}: ${(await response.text()).slice(0, 500)}`);
  const data = await response.json() as any;
  return String(data?.choices?.[0]?.message?.content || '').trim() || 'Не удалось получить ответ от AI.';
}

async function handleTelegram(request: Request, env: Env, ctx: ExecutionContext) {
  if (env.TELEGRAM_WEBHOOK_SECRET) {
    const supplied = request.headers.get('X-Telegram-Bot-Api-Secret-Token');
    if (supplied !== env.TELEGRAM_WEBHOOK_SECRET) return json({ ok: false }, 401);
  }

  const update = await request.json() as any;
  const message = update?.message;
  const chatId = message?.chat?.id;
  const text = typeof message?.text === 'string' ? message.text.trim() : '';
  if (!chatId || !text) return json({ ok: true, ignored: true });

  ctx.waitUntil((async () => {
    try {
      const reply = await generateReply(env, text);
      await telegram(env, 'sendMessage', { chat_id: chatId, text: reply });
    } catch (error) {
      console.error('[Telegram]', error);
      await telegram(env, 'sendMessage', {
        chat_id: chatId,
        text: 'Не удалось получить ответ AI. Попробуй ещё раз.',
      }).catch(() => undefined);
    }
  })());

  return json({ ok: true });
}

async function handleChat(request: Request, env: Env) {
  const body = await request.json() as any;
  const text = String(body?.message ?? body?.text ?? '').trim();
  if (!text) return json({ error: 'message is required' }, 400, cors(env));
  try {
    const reply = await generateReply(env, text, Array.isArray(body?.history) ? body.history : []);
    return json({ reply, text: reply }, 200, cors(env));
  } catch (error) {
    console.error('[Chat]', error);
    return json({ error: 'AI provider error' }, 502, cors(env));
  }
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors(env) });
    if (url.pathname === '/health') return json({ ok: true, service: 'xori-cloudflare-worker' });
    if (url.pathname === '/api/chat' && request.method === 'POST') return handleChat(request, env);
    if (url.pathname === '/telegram/webhook' && request.method === 'POST') return handleTelegram(request, env, ctx);
    return json({ ok: true, service: 'xori-cloudflare-worker', endpoints: ['/health', '/api/chat', '/telegram/webhook'] });
  },
};
