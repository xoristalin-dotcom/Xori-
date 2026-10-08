import React, { useState, useEffect } from 'react';
import { MessageSquare, BookOpen, Brain, Compass, Settings2, Heart } from 'lucide-react';
import { ThreeViewer } from './components/ThreeViewer';
import { ControlView } from './components/ControlView';
import { ChatInterface } from './components/ChatInterface';
import { DiaryView } from './components/DiaryView';
import { MemoryView } from './components/MemoryView';
import { PersonalityKnowledgeView } from './components/PersonalityKnowledgeView';
import { ChatMessage, AnimationType, EmotionType, DiaryEntry, UserMemory, PersonalityData, KnowledgeData } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'diary' | 'memory' | 'personality' | 'control'>('chat');
  const [currentAnimation, setCurrentAnimation] = useState<AnimationType>('idle');
  const [currentEmotion, setCurrentEmotion] = useState<EmotionType>('calm');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [diaryEntries, setDiaryEntries] = useState<DiaryEntry[]>([]);
  const [memory, setMemory] = useState<UserMemory>({ user_name: 'мой любимый', facts: [], interests: [], conversations: [], mood: 'спокойное', emotion: 'calm', last_interaction: null, proactive_sent: [] });
  const [personality, setPersonality] = useState<PersonalityData | null>(null);
  const [knowledge, setKnowledge] = useState<KnowledgeData | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [memRes, diaryRes, personRes, knowRes] = await Promise.all([
          fetch(`${API_BASE}/api/memory`).then(r => r.json()).catch(() => null),
          fetch(`${API_BASE}/api/diary`).then(r => r.json()).catch(() => null),
          fetch(`${API_BASE}/api/personality`).then(r => r.json()).catch(() => null),
          fetch(`${API_BASE}/api/knowledge`).then(r => r.json()).catch(() => null),
        ]);
        if (memRes) { setMemory(memRes); if (memRes.emotion) setCurrentEmotion(memRes.emotion); }
        if (diaryRes?.entries) setDiaryEntries(diaryRes.entries);
        if (personRes) setPersonality(personRes);
        if (knowRes) setKnowledge(knowRes);
        setMessages([{ id: 'init-1', sender: 'hori', text: `Привет! Я Хори Кёко. ${memRes?.user_name ? `${memRes.user_name}, рада тебя видеть!` : 'Рада тебя видеть!'} Как твой день проходит?`, time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }), emotion: 'happy', animation: 'wave' }]);
        setCurrentAnimation('wave');
        setTimeout(() => setCurrentAnimation('idle'), 4000);
      } catch (err) { console.error('Error loading initial data:', err); }
    };
    fetchData();
  }, []);

  const handleSendMessage = async (text: string) => {
    const userMsg: ChatMessage = { id: Date.now().toString(), sender: 'user', text, time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) };
    setMessages(prev => [...prev, userMsg]);
    setIsGenerating(true);
    try {
      const res = await fetch(`${API_BASE}/api/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text, history: messages.slice(-8) }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'API-провайдеры недоступны');
      const horiMsg: ChatMessage = { id: (Date.now() + 1).toString(), sender: 'hori', text: data.reply || 'Я здесь, всё хорошо!', time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }), emotion: data.emotion || 'calm', animation: data.animation || 'idle' };
      setMessages(prev => [...prev, horiMsg]);
      if (data.emotion) setCurrentEmotion(data.emotion);
      if (data.animation) setCurrentAnimation(data.animation);
      if (data.memory) setMemory(data.memory);
      if (data.diaryEntries) setDiaryEntries(data.diaryEntries);
    } catch (err) {
      console.error('Chat error:', err);
      setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), sender: 'hori', text: err instanceof Error ? `Не могу ответить через API: ${err.message}` : 'Не могу ответить через API: провайдеры недоступны.', time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }), emotion: 'thinking' }]);
    } finally { setIsGenerating(false); }
  };

  const handleGenerateDiaryThought = async () => {
    setIsGenerating(true);
    try { const res = await fetch(`${API_BASE}/api/diary/generate`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages: messages.slice(-10) }) }); const data = await res.json(); if (data.entries) setDiaryEntries(data.entries); setCurrentAnimation('happy'); setTimeout(() => setCurrentAnimation('idle'), 4000); } catch (e) { console.error('Diary error:', e); } finally { setIsGenerating(false); }
  };

  return <div className="min-h-screen"><ThreeViewer animation={currentAnimation} emotion={currentEmotion} /><ChatInterface messages={messages} onSend={handleSendMessage} isGenerating={isGenerating} /></div>;
};
