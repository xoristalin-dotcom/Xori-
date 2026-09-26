import fs from 'node:fs';

const path = 'server.ts';
if (!fs.existsSync(path)) process.exit(0);

let s = fs.readFileSync(path, 'utf8');

if (!s.includes("const PROACTIVE_CHAT_ID")) {
  s = s.replace(
    "const CONTROL_TOKEN = process.env.HORI_CONTROL_TOKEN || '';",
    "const CONTROL_TOKEN = process.env.HORI_CONTROL_TOKEN || '';\nconst PROACTIVE_CHAT_ID = Number(process.env.HORI_PROACTIVE_CHAT_ID || 0);",
  );
}

s = s.replace(
  "  next.last_chat_id = next.last_chat_id ?? null;",
  "  next.last_chat_id = next.last_chat_id ?? (PROACTIVE_CHAT_ID || null);",
);

if (!s.includes("[Telegram] chat_id=")) {
  s = s.replace(
    "  if (!chatId || !text) return;",
    "  if (!chatId || !text) return;\n  console.log('[Telegram] chat_id=' + String(chatId));",
  );
}

fs.writeFileSync(path, s, 'utf8');
console.log('Proactive persistence patch applied.');
