import React, { useState, useRef, useEffect } from 'react';
import { Send, Volume2, VolumeX, Sparkles, Trash2, Heart, MessageSquare } from 'lucide-react';
import { ChatMessage, EmotionType, AnimationType } from '../types';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => Promise<void>;
  isGenerating: boolean;
  onAnimationTrigger: (anim: AnimationType) => void;
  onClearChat: () => void;
  currentEmotion: EmotionType;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  isGenerating,
  onAnimationTrigger,
  onClearChat,
  currentEmotion,
}) => {
  const [inputText, setInputText] = useState('');
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const lastSpokenMessageId = useRef<string | null>(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  // Speech synthesis for Hori's messages
  const speakText = (text: string) => {
    if (!voiceEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ru-RU';
    utterance.pitch = 1.15; // slightly higher pitch for anime-like warmth
    utterance.rate = 1.05;

    const voices = window.speechSynthesis.getVoices();
    const ruFemaleVoice = voices.find(
      (v) => v.lang.startsWith('ru') && (v.name.includes('Female') || v.name.includes('Milena') || v.name.includes('Tatyana') || v.name.includes('Google русский'))
    ) || voices.find((v) => v.lang.startsWith('ru'));

    if (ruFemaleVoice) {
      utterance.voice = ruFemaleVoice;
    }
    window.speechSynthesis.speak(utterance);
  };

  useEffect(() => {
    const latestMessage = messages[messages.length - 1];
    if (!latestMessage || latestMessage.sender !== 'hori' || latestMessage.id === lastSpokenMessageId.current) return;
    lastSpokenMessageId.current = latestMessage.id;
    speakText(latestMessage.text);
  }, [messages]);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed || isGenerating) return;

    setInputText('');
    await onSendMessage(trimmed);
  };

  // Emotion labels
  const emotionLabels: Record<EmotionType, { label: string; color: string; icon: string }> = {
    calm: { label: 'Спокойна', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', icon: '🍃' },
    happy: { label: 'Радостная', color: 'bg-amber-500/20 text-amber-300 border-amber-500/40', icon: '✨' },
    thinking: { label: 'Задумалась', color: 'bg-sky-500/20 text-sky-300 border-sky-500/40', icon: '💭' },
    sad: { label: 'Грустит', color: 'bg-blue-500/20 text-blue-300 border-blue-500/40', icon: '🌧️' },
    angry: { label: 'Вспылила', color: 'bg-rose-500/20 text-rose-300 border-rose-500/40', icon: '💢' },
    wave: { label: 'Приветствует', color: 'bg-purple-500/20 text-purple-300 border-purple-500/40', icon: '👋' },
    dance: { label: 'Танцует', color: 'bg-pink-500/20 text-pink-300 border-pink-500/40', icon: '💃' },
  };

  const quickPrompts = [
    'Привет, Хори! Чем занимаешься?',
    'Расскажи о себе и Миямуре',
    'Что ты готовишь сегодня на ужин?',
    'Потанцуй для меня!',
  ];

  return (
    <div
      id="chat-interface-panel"
      className="flex flex-col h-full bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl overflow-hidden backdrop-blur-md"
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-800/80 flex items-center justify-between bg-slate-950/40">
        <div className="flex items-center gap-2.5">
          <div className="relative">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-rose-500 to-amber-400 flex items-center justify-center font-bold text-white shadow-md shadow-rose-500/20">
              Х
            </div>
            <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-400 border-2 border-slate-900 rounded-full" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-slate-100">Хори Кёко</h2>
              <span className={`text-[11px] px-2 py-0.5 rounded-full border ${emotionLabels[currentEmotion]?.color || 'border-slate-700 text-slate-300'}`}>
                {emotionLabels[currentEmotion]?.icon} {emotionLabels[currentEmotion]?.label}
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Horimiya • Старшая школа Катагири</p>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            id="toggle-voice-btn"
            onClick={() => setVoiceEnabled(!voiceEnabled)}
            title={voiceEnabled ? 'Озвучка включена' : 'Озвучка выключена'}
            className={`p-2 rounded-xl border transition-all ${
              voiceEnabled
                ? 'bg-rose-500/20 border-rose-500/50 text-rose-300'
                : 'bg-slate-800/80 border-slate-700 text-slate-400 hover:text-slate-200'
            }`}
          >
            {voiceEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>

          <button
            id="clear-chat-btn"
            onClick={onClearChat}
            title="Очистить переписку"
            className="p-2 rounded-xl bg-slate-800/80 border border-slate-700 text-slate-400 hover:text-rose-400 transition-all"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3.5">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-400">
            <div className="w-14 h-14 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400 mb-3">
              <Heart className="w-7 h-7" />
            </div>
            <h3 className="text-sm font-semibold text-slate-200 mb-1">Хори здесь и слушает тебя</h3>
            <p className="text-xs text-slate-400 max-w-xs mb-4">
              Напиши ей что угодно: о делах, школе, готовке или просто спроси как её день!
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-2.5 shadow-md text-sm leading-relaxed ${
                  msg.sender === 'user'
                    ? 'bg-rose-600 text-white rounded-tr-none'
                    : 'bg-slate-800/90 border border-slate-700/70 text-slate-200 rounded-tl-none'
                }`}
              >
                <div className="whitespace-pre-wrap">{msg.text}</div>

                {/* Footer of message */}
                <div className="mt-1 flex items-center justify-between gap-3 text-[10px] opacity-70">
                  <span>{msg.time}</span>
                  {msg.sender === 'hori' && (
                    <div className="flex items-center gap-1.5">
                      {msg.animation && (
                        <button
                          onClick={() => onAnimationTrigger(msg.animation!)}
                          className="hover:underline text-rose-300 flex items-center gap-0.5"
                        >
                          <Sparkles className="w-2.5 h-2.5" />
                          <span>{msg.animation}</span>
                        </button>
                      )}
                      <button
                        onClick={() => speakText(msg.text)}
                        title="Прослушать"
                        className="hover:text-rose-300 p-0.5"
                      >
                        <Volume2 className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))
        )}

        {isGenerating && (
          <div className="flex items-start gap-2">
            <div className="bg-slate-800/90 border border-slate-700/70 rounded-2xl rounded-tl-none px-4 py-3 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-rose-400 animate-bounce" />
              <div className="w-2 h-2 rounded-full bg-rose-400 animate-bounce [animation-delay:0.2s]" />
              <div className="w-2 h-2 rounded-full bg-rose-400 animate-bounce [animation-delay:0.4s]" />
              <span className="text-xs text-slate-400 ml-1">Хори печатает...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick Prompts */}
      <div className="px-3 py-2 border-t border-slate-800/60 bg-slate-950/20 flex gap-2 overflow-x-auto no-scrollbar">
        {quickPrompts.map((prompt, i) => (
          <button
            key={i}
            onClick={() => onSendMessage(prompt)}
            disabled={isGenerating}
            className="whitespace-nowrap px-3 py-1 rounded-full text-xs bg-slate-800/70 hover:bg-rose-500/20 hover:text-rose-200 border border-slate-700/60 text-slate-300 transition-all disabled:opacity-50"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Input Form */}
      <form
        onSubmit={handleSend}
        className="p-3 bg-slate-950/60 border-t border-slate-800 flex items-center gap-2"
      >
        <input
          id="chat-user-input"
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Напиши Хори..."
          disabled={isGenerating}
          className="flex-1 bg-slate-800/80 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-rose-500 focus:ring-1 focus:ring-rose-500 transition-all disabled:opacity-50"
        />
        <button
          id="chat-submit-btn"
          type="submit"
          disabled={!inputText.trim() || isGenerating}
          className="p-2.5 rounded-xl bg-rose-500 hover:bg-rose-600 disabled:opacity-50 text-white shadow-md shadow-rose-500/30 transition-all cursor-pointer disabled:cursor-not-allowed"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
};
