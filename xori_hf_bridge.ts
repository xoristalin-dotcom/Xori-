type HistoryItem = {
  sender?: string;
  text?: string;
  role?: string;
  content?: string;
};

type HfReply = {
  text: string;
  model: string;
};

const SPACE_URL = (process.env.XORI_HF_SPACE_URL || '').replace(/\/$/, '');
const BRIDGE_TOKEN = process.env.XORI_HF_BRIDGE_TOKEN || '';
const ENDPOINT = process.env.XORI_HF_ENDPOINT || 'generate';
const TIMEOUT_MS = Number.parseInt(process.env.XORI_HF_TIMEOUT_MS || '90000', 10);

function normalizeHistory(history: HistoryItem[]): Array<{ role: 'user' | 'assistant'; content: string }> {
  return history.slice(-8).map((item) => ({
    role: item.role === 'assistant' || item.sender === 'hori' ? 'assistant' : 'user',
    content: String(item.content ?? item.text ?? '').trim(),
  })).filter((item) => item.content);
}

function extractEventId(body: any): string {
  if (typeof body === 'string') return body.trim();
  return String(body?.event_id || body?.eventId || '').trim();
}

async function readSseResult(response: Response): Promise<any> {
  const text = await response.text();
  const lines = text.split(/\r?\n/);
  let currentEvent = '';
  for (const line of lines) {
    if (line.startsWith('event:')) currentEvent = line.slice(6).trim();
    if (line.startsWith('data:')) {
      const raw = line.slice(5).trim();
      if (!raw || raw === '[DONE]') continue;
      try {
        const parsed = JSON.parse(raw);
        if (currentEvent === 'complete' || currentEvent === 'generating' || parsed?.error) return parsed;
      } catch {
        // Ignore keep-alive/non-JSON SSE frames.
      }
    }
  }
  return null;
}

async function callGradioSpace(systemPrompt: string, message: string, history: HistoryItem[]): Promise<string> {
  if (!SPACE_URL) throw new Error('XORI_HF_SPACE_URL is not configured');

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (BRIDGE_TOKEN) headers.Authorization = `Bearer ${BRIDGE_TOKEN}`;

  const prompt = [
    systemPrompt,
    '',
    'Conversation history:',
    JSON.stringify(normalizeHistory(history)),
    '',
    'User:',
    message,
  ].join('\n');

  const start = await fetch(`${SPACE_URL}/gradio_api/call/${ENDPOINT}`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ data: [prompt] }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });

  if (!start.ok) {
    const body = await start.text();
    throw new Error(`HF Space start failed (${start.status}): ${body.slice(0, 400)}`);
  }

  const eventId = extractEventId(await start.json());
  if (!eventId) throw new Error('HF Space did not return an event id');

  const result = await fetch(`${SPACE_URL}/gradio_api/call/${ENDPOINT}/${encodeURIComponent(eventId)}`, {
    headers,
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });

  if (!result.ok) {
    const body = await result.text();
    throw new Error(`HF Space result failed (${result.status}): ${body.slice(0, 400)}`);
  }

  const payload = await readSseResult(result);
  if (payload?.error) throw new Error(String(payload.error));
  const value = Array.isArray(payload?.data) ? payload.data[0] : payload?.data;
  if (typeof value !== 'string' || !value.trim()) throw new Error('HF Space returned no text');
  return value.trim();
}

export async function generateXoriHfReply(
  systemPrompt: string,
  userText: string,
  history: HistoryItem[] = [],
): Promise<HfReply | null> {
  if (!SPACE_URL || !userText.trim()) return null;
  const text = await callGradioSpace(systemPrompt, userText.trim(), history);
  return {
    text,
    model: process.env.XORI_HF_MODEL_LABEL || 'Xori on Hugging Face Space',
  };
}

export function getXoriHfBridgeStatus() {
  return {
    enabled: Boolean(SPACE_URL),
    spaceUrlConfigured: Boolean(SPACE_URL),
    endpoint: ENDPOINT,
    model: process.env.XORI_HF_MODEL_ID || 'Qwen/Qwen2.5-0.5B-Instruct',
    timeoutMs: TIMEOUT_MS,
    authConfigured: Boolean(BRIDGE_TOKEN),
    role: 'Render bridge -> Hugging Face Space',
  };
}
