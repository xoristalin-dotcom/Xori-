import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'path'; // <--- Добавляем импорт пути

export default defineConfig({
  plugins: [react(), tailwindcss()],
  
  // 1. Настройка алиасов для удобных импортов (опционально, но очень удобно)
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },

  // 2. Настройки сервера разработки (ваши текущие сохранены)
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: true,
    // (Полезно) Если бэкенд (server.ts) запущен на другом порту (например, 5000), 
    // можно добавить прокси, чтобы не было проблем с CORS при запросах:
    // proxy: {
    //   '/api': 'http://localhost:5000',
    // },
  },

  // 3. Оптимизация сборки для продакшена
  build: {
    outDir: 'dist',
    sourcemap: false, // Отключаем карты кода в продакшене для безопасности и меньшего веса
    // Улучшаем кэширование и разделение вендорных библиотек
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
        },
      },
    },
  },
});

