import React, { useEffect, useMemo, useState } from 'react';
import { Activity, Bot, CheckCircle2, Clock3, Image, Mic, RefreshCw, Send, Settings2, Shield, Sparkles, Zap } from 'lucide-react';

type ControlState = {
  enabled: boolean;
  proactiveEnabled: boolean;
  voiceEnabled: boolean;
  photoEnabled: boolean;
  autoFirst: boolean;
  modelMode: string;
  replyDelayMode: 'instant' | 'fixed' | 'random';
  replyDelayHours: number;
  replyDelayMinHours: number;
  replyDelayMaxHours: number;
  updatedAt?: string | null;
  tokenConfigured?: boolean;
};

type AutonomyState = {
  autonomy?: boolean;
  proactive?: boolean;
  next_proactive_at?: string | null;
  last_proactive_message?: string | null;
  conversation_state?: string;
  telegram_voice_configured?: boolean;
  pollinations_image_configured?: boolean;
};

const defaults: ControlState = {
  enabled: true,
  proactiveEnabled: true,
  voiceEnabled: true,
  photoEnabled: true,
  autoFirst: true,
  modelMode: 'auto',
  replyDelayMode: 'instant',
  replyDelayHours: 0,
  replyDelayMinHours: 0,
  replyDelayMaxHours: 6,
};

export const ControlView: React.FC = () => {
  const [control, setControl] = useState<ControlState>(defaults);
  const [autonomy, setAutonomy] = useState<AutonomyState>({});
  const [token, setToken] = useState(() => sessionStorage.getItem('hori-control-token') || '');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [testChatId, setTestChatId] = useState('');
  const [testText, setTestText] = useState('Привет! Это тестовое сообщение от Hori Control.');
  const [testStatus, setTestStatus] = useState('');

  const headers = useMemo(() => ({
    'Content-Type': 'application/json',
    ...(token ? { 'x-hori-control-token': token } : {}),
  }), [token]);

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const [controlRes, autonomyRes] = await Promise.all([
        fetch('/api/control'),
        fetch('/api/autonomy'),
      ]);
      const controlData = await controlRes.json();
      const autonomyData = await autonomyRes.json();
      if (controlData?.ok) setControl({ ...defaults, ...controlData });
      if (autonomyRes.ok) setAutonomy(autonomyData);
    } catch (e) {
      setError('Не удалось получить состояние бота.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const patch = async (next: Partial<ControlState>) => {
    setSaving(true);
    setError('');
    try {
      const res = await fetch('/api/control', {
        method: 'POST',
        headers,
        body: JSON.stringify(next),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || 'Не удалось сохранить настройки');
      setControl({ ...defaults, ...data });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ошибка сохранения');
    } finally {
      setSaving(false);
    }
  };

  const setTokenAndSave = (value: string) => {
    setToken(value);
    sessionStorage.setItem('hori-control-token', value);
  };

  const sendTest = async () => {
    setTestStatus('');
    try {
      const res = await fetch('/api/control/test-message', {
        method: 'POST',
        headers,
        body: JSON.stringify({ chatId: testChatId, text: testText }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || 'Ошибка отправки');
      setTestStatus('Тестовое сообщение отправлено.');
    } catch (e) {
      setTestStatus(e instanceof Error ? e.message : 'Не удалось отправить сообщение.');
    }
  };

  const toggle = (key: keyof ControlState) => {
    patch({ [key]: !control[key] } as Partial<ControlState>);
  };

  return (
    <div className="h-full overflow-y-auto rounded-2xl border border-slate-800 bg-slate-950/70 p-4 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-rose-300" />
            <h2 className="font-bold">Управление ботом</h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">Автономность, медиа, задержки и состояние сервисов.</p>
        </div>
        <button onClick={load} className="p-2 rounded-lg bg-slate-900 border border-slate-800 hover:bg-slate-800" title="Обновить">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {error && <div className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded-lg p-2">{error}</div>}

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        <Status icon={<Bot />} label="Бот" ok={!!autonomy.autonomy} />
        <Status icon={<Zap />} label="Автономность" ok={!!autonomy.proactive} />
        <Status icon={<Mic />} label="Voice" ok={!!autonomy.telegram_voice_configured} />
        <Status icon={<Image />} label="Photo" ok={!!autonomy.pollinations_image_configured} />
        <Status icon={<Shield />} label="Токен" ok={!!control.tokenConfigured} />
        <Status icon={<Activity />} label="Работает" ok={control.enabled} />
      </div>

      <Section title="Автономность">
        <Toggle label="Бот включён" value={control.enabled} onChange={() => toggle('enabled')} />
        <Toggle label="Может писать сам" value={control.proactiveEnabled} onChange={() => toggle('proactiveEnabled')} />
        <Toggle label="Может начать первым" value={control.autoFirst} onChange={() => toggle('autoFirst')} />
      </Section>

      <Section title="Медиа">
        <Toggle label="Голосовые сообщения" value={control.voiceEnabled} onChange={() => toggle('voiceEnabled')} />
        <Toggle label="Фотографии" value={control.photoEnabled} onChange={() => toggle('photoEnabled')} />
      </Section>

      <Section title="Ответы">
        <label className="block text-xs text-slate-400 mb-1">Режим задержки</label>
        <select
          value={control.replyDelayMode}
          onChange={(e) => patch({ replyDelayMode: e.target.value as ControlState['replyDelayMode'] })}
          className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm"
        >
          <option value="instant">Мгновенно</option>
          <option value="fixed">Фиксированная задержка</option>
          <option value="random">Случайная задержка</option>
        </select>

        {control.replyDelayMode === 'fixed' && (
          <NumberField label="Задержка, часов" value={control.replyDelayHours} onChange={(v) => patch({ replyDelayHours: v })} />
        )}

        {control.replyDelayMode === 'random' && (
          <div className="grid grid-cols-2 gap-2">
            <NumberField label="Минимум, ч" value={control.replyDelayMinHours} onChange={(v) => patch({ replyDelayMinHours: v })} />
            <NumberField label="Максимум, ч" value={control.replyDelayMaxHours} onChange={(v) => patch({ replyDelayMaxHours: v })} />
          </div>
        )}
      </Section>

      <Section title="Модель">
        <label className="block text-xs text-slate-400 mb-1">Режим модели</label>
        <select
          value={control.modelMode}
          onChange={(e) => patch({ modelMode: e.target.value })}
          className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm"
        >
          <option value="auto">Авто + fallback</option>
          <option value="xori">Xori</option>
          <option value="pollinations">Pollinations</option>
        </select>
      </Section>

      <Section title="Тест Telegram">
        <input value={testChatId} onChange={(e) => setTestChatId(e.target.value)} placeholder="Chat ID" className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm mb-2" />
        <textarea value={testText} onChange={(e) => setTestText(e.target.value)} rows={2} className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm mb-2" />
        <button disabled={saving || !testChatId || !token} onClick={sendTest} className="w-full py-2 rounded-lg bg-rose-500 hover:bg-rose-400 disabled:opacity-40 font-semibold text-sm flex items-center justify-center gap-2">
          <Send className="w-4 h-4" /> Отправить тест
        </button>
        {testStatus && <p className="text-xs text-slate-400 mt-2">{testStatus}</p>}
      </Section>

      <Section title="Доступ">
        <p className="text-xs text-slate-500 mb-2">Токен хранится только в текущей сессии браузера.</p>
        <input type="password" value={token} onChange={(e) => setTokenAndSave(e.target.value)} placeholder="HORI_CONTROL_TOKEN" className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm" />
      </Section>

      <div className="text-[11px] text-slate-500 flex items-center gap-2">
        <Clock3 className="w-3.5 h-3.5" />
        {autonomy.next_proactive_at ? `Следующее автономное действие: ${new Date(autonomy.next_proactive_at).toLocaleString('ru-RU')}` : 'Следующее автономное действие пока не запланировано'}
      </div>

      {saving && <div className="fixed bottom-4 right-4 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs flex items-center gap-2"><Sparkles className="w-3.5 h-3.5" /> Сохраняю…</div>}
    </div>
  );
};

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-3 space-y-3">
    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">{title}</h3>
    {children}
  </section>
);

const Toggle: React.FC<{ label: string; value: boolean; onChange: () => void }> = ({ label, value, onChange }) => (
  <button onClick={onChange} className="w-full flex items-center justify-between gap-3 text-left">
    <span className="text-sm">{label}</span>
    <span className={`w-10 h-5 rounded-full p-0.5 transition-colors ${value ? 'bg-rose-500' : 'bg-slate-700'}`}>
      <span className={`block w-4 h-4 rounded-full bg-white transition-transform ${value ? 'translate-x-5' : ''}`} />
    </span>
  </button>
);

const NumberField: React.FC<{ label: string; value: number; onChange: (value: number) => void }> = ({ label, value, onChange }) => (
  <label className="text-xs text-slate-400">
    {label}
    <input type="number" min="0" max="6" step="0.1" value={value} onChange={(e) => onChange(Number(e.target.value))} className="mt-1 w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100" />
  </label>
);

const Status: React.FC<{ icon: React.ReactNode; label: string; ok: boolean }> = ({ icon, label, ok }) => (
  <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-2 flex items-center gap-2">
    <span className={ok ? 'text-emerald-400' : 'text-slate-600'}>{React.cloneElement(icon as React.ReactElement, { className: 'w-3.5 h-3.5' })}</span>
    <span className="text-[11px] text-slate-300 truncate">{label}</span>
    <CheckCircle2 className={`ml-auto w-3 h-3 ${ok ? 'text-emerald-400' : 'text-slate-700'}`} />
  </div>
);
