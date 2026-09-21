import React, { useState, useEffect } from 'react';
import { MessageSquare, BookOpen, Brain, Compass, Sparkles, Heart, RefreshCw } from 'lucide-react';
import { ThreeViewer } from './components/ThreeViewer';
import { ChatInterface } from './components/ChatInterface';
import { DiaryView } from './components/DiaryView';
import { MemoryView } from './components/MemoryView';
import { PersonalityKnowledgeView } from './components/PersonalityKnowledgeView';
import {
  ChatMessage,
  AnimationType,
  EmotionType,
  DiaryEntry,
  UserMemory,
  PersonalityData,
  KnowledgeData,
} from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'diary' | 'memory' | 'personality'>('chat');
  const [currentAnimation, setCurrentAnimation] = useState<AnimationType>('idle');
  const [currentEmotion, setCurrentEmotion] = useState<EmotionType>('calm');

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const [diaryEntries, setDiaryEntries] = useState<DiaryEntry[]>([]);
  const [memory, setMemory] = useState<UserMemory>({
    user_name: 'мой любимый',
    facts: [],
    interests: [],
    conversations: [],
    mood: 'спокойное',
    emotion: 'calm',
    last_interaction: null,
    proactive_sent: [],
  });
  const [personality, setPersonality] = useState<PersonalityData | null>(null);
  const [knowledge, setKnowledge] = useState<KnowledgeData | null>(null);

  // Fetch initial data from backend API
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [memRes, diaryRes, personRes, knowRes] = await Promise.all([
          fetch('/api/memory').then((r) => r.json()),
          fetch('/api/diary').then((r) => r.json()),
          fetch('/api/personality').then((r) => r.json()),
          fetch('/api/knowledge').then((r) => r.json()),
        ]);

        if (memRes) {
          setMemory(memRes);
          if (memRes.emotion) setCurrentEmotion(memRes.emotion);
        }
        if (diaryRes?.entries) {
          setDiaryEntries(diaryRes.entries);
        }
        if (personRes) setPersonality(personRes);
        if (knowRes) setKnowledge(knowRes);

        // Initial welcome message from Hori
        setMessages([
          {
            id: 'init-1',
            sender: 'hori',
            text: `Привет! Я Хори Кёко. ${memRes?.user_name ? `${memRes.user_name}, рада тебя видеть!` : 'Рада тебя видеть!'} Как твой день проходит?`,
            time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
            emotion: 'happy',
            animation: 'wave',
          },
        ]);
        setCurrentAnimation('wave');
        setTimeout(() => setCurrentAnimation('idle'), 4000);
      } catch (err) {
        console.error('Error loading initial data:', err);
      }
    };

    fetchData();
  }, []);

  // Send message to Hori
  const handleSendMessage = async (text: string) => {
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text,
      time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsGenerating(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history: messages.slice(-8),
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'API-провайдеры недоступны');
      }

      const horiMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'hori',
        text: data.reply || 'Я здесь, всё хорошо!',
        time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
        emotion: data.emotion || 'calm',
        animation: data.animation || 'idle',
      };

      setMessages((prev) => [...prev, horiMsg]);

      if (data.emotion) {
        setCurrentEmotion(data.emotion);
      }

      if (data.animation) {
        setCurrentAnimation(data.animation);
      }

      // Update memory if refreshed
      if (data.memory) {
        setMemory(data.memory);
      }

      // Update diary if new entry was added
      if (data.diaryEntries) {
        setDiaryEntries(data.diaryEntries);
      }
    } catch (err) {
      console.error('Chat error:', err);
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'hori',
        text: err instanceof Error
          ? `Не могу ответить через API: ${err.message}`
          : 'Не могу ответить через API: провайдеры недоступны.',
        time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
        emotion: 'thinking',
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsGenerating(false);
    }
  };

  // Trigger Hori to write a diary entry
  const handleGenerateDiaryThought = async () => {
    setIsGenerating(true);
    try {
      const res = await fetch('/api/diary/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: messages.slice(-10) }),
      });
      const data = await res.json();
      if (data.entries) {
        setDiaryEntries(data.entries);
      }
      setCurrentAnimation('happy');
      setTimeout(() => setCurrentAnimation('idle'), 4000);
    } catch (e) {
      console.error('Diary error:', e);
    } finally {
      setIsGenerating(false);
    }
  };

  // Add custom diary entry
  const handleAddDiaryEntry = async (entry: Omit<DiaryEntry, 'id'>) => {
    try {
      const res = await fetch('/api/diary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entry),
      });
      const data = await res.json();
      if (data.entries) {
        setDiaryEntries(data.entries);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Add fact to memory
  const handleAddFact = async (factText: string) => {
    try {
      const res = await fetch('/api/memory/facts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fact: factText }),
      });
      const data = await res.json();
      if (data.memory) {
        setMemory(data.memory);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Remove fact from memory
  const handleRemoveFact = async (index: number) => {
    try {
      const res = await fetch(`/api/memory/facts/${index}`, {
        method: 'DELETE',
      });
      const data = await res.json();
      if (data.memory) {
        setMemory(data.memory);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Update user name in memory
  const handleUpdateUserName = async (name: string) => {
    try {
      const res = await fetch('/api/memory/user_name', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_name: name }),
      });
      const data = await res.json();
      if (data.memory) {
        setMemory(data.memory);
      }
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="min-h-screen bg-[#0d0f17] text-slate-100 flex flex-col antialiased">
      {/* Top Navbar */}
      <header className="border-b border-slate-800/80 bg-slate-950/60 backdrop-blur-md sticky top-0 z-50 px-4 lg:px-8 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-rose-500 to-amber-400 flex items-center justify-center font-bold text-white shadow-lg shadow-rose-500/25">
            <Heart className="w-5 h-5 fill-white" />
          </div>
          <div>
            <h1 className="text-sm lg:text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
              <span>Хори Кёко</span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-300 border border-rose-500/30">
                3D ИИ-помощник
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">Horimiya • Интерактивный компаньон</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-1.5 bg-slate-900/90 border border-slate-800 p-1 rounded-xl shadow-inner">
          <button
            id="tab-chat"
            onClick={() => setActiveTab('chat')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              activeTab === 'chat'
                ? 'bg-rose-500 text-white shadow-sm shadow-rose-500/40'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>Диалог</span>
          </button>

          <button
            id="tab-diary"
            onClick={() => setActiveTab('diary')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              activeTab === 'diary'
                ? 'bg-rose-500 text-white shadow-sm shadow-rose-500/40'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Дневник</span>
            {diaryEntries.length > 0 && (
              <span className="text-[10px] px-1.5 py-0.2 bg-slate-800 rounded-full text-slate-300 font-normal">
                {diaryEntries.length}
              </span>
            )}
          </button>

          <button
            id="tab-memory"
            onClick={() => setActiveTab('memory')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              activeTab === 'memory'
                ? 'bg-rose-500 text-white shadow-sm shadow-rose-500/40'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Brain className="w-3.5 h-3.5" />
            <span>Память</span>
            {memory.facts?.length > 0 && (
              <span className="text-[10px] px-1.5 py-0.2 bg-slate-800 rounded-full text-slate-300 font-normal">
                {memory.facts.length}
              </span>
            )}
          </button>

          <button
            id="tab-personality"
            onClick={() => setActiveTab('personality')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              activeTab === 'personality'
                ? 'bg-rose-500 text-white shadow-sm shadow-rose-500/40'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Канон</span>
          </button>
        </div>
      </header>

      {/* Main Responsive Grid: 3D Stage on left, Companion Panels on right */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-3 sm:p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-6 min-h-0">
        {/* Left: 3D Character Viewport */}
        <div className="lg:col-span-6 xl:col-span-7 h-[420px] lg:h-[calc(100vh-130px)] min-h-[400px]">
          <ThreeViewer
            currentAnimation={currentAnimation}
            onAnimationChange={(anim) => setCurrentAnimation(anim)}
            emotion={currentEmotion}
          />
        </div>

        {/* Right: Companion Controls & Tabs */}
        <div className="lg:col-span-6 xl:col-span-5 h-[520px] lg:h-[calc(100vh-130px)] min-h-[460px]">
          {activeTab === 'chat' && (
            <ChatInterface
              messages={messages}
              onSendMessage={handleSendMessage}
              isGenerating={isGenerating}
              onAnimationTrigger={(anim) => setCurrentAnimation(anim)}
              onClearChat={() => setMessages([])}
              currentEmotion={currentEmotion}
            />
          )}

          {activeTab === 'diary' && (
            <DiaryView
              entries={diaryEntries}
              onAddEntry={handleAddDiaryEntry}
              onGenerateDiaryThought={handleGenerateDiaryThought}
              isGenerating={isGenerating}
            />
          )}

          {activeTab === 'memory' && (
            <MemoryView
              memory={memory}
              onAddFact={handleAddFact}
              onRemoveFact={handleRemoveFact}
              onUpdateUserName={handleUpdateUserName}
            />
          )}

          {activeTab === 'personality' && (
            <PersonalityKnowledgeView
              personality={personality}
              knowledge={knowledge}
            />
          )}
        </div>
      </main>
    </div>
  );
};
