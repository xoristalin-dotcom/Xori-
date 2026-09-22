# Хори Кёко — 3D ИИ-помощник

## Локальный запуск

```bash
npm install
npm run dev
```

### Резервные API и офлайн-режим

Сервер на каждом сообщении пробует настроенные Gemini, OpenRouter, Groq, Kie.ai и OpenAI по цепочке. Если один ключ получил `429`, истёк или временно недоступен, запрос автоматически переходит к следующему. Когда все провайдеры недоступны, используется внутренний генератор Хори.

Проверить конфигурацию без раскрытия ключей можно через `GET /api/providers`. Поле `configured` показывает, загружен ли ключ, а `coolingDown` — временно ли провайдер отключён после ошибки. Во время локальной проверки ключ `GEMINI_API_KEY` вернул `API_KEY_INVALID`, поэтому его нужно заменить в `.env` или Secrets Render.

Для полного офлайн-режима можно запустить сервер с `OFFLINE_MODE=true npm run dev`. В обычном режиме оставляй `OFFLINE_MODE=false` или не задавай переменную, чтобы доступные API использовались автоматически.

## Деплой на Render.com

1. Зарегистрируйся на [render.com](https://render.com)
2. Создай новый **Web Service** → выбери GitHub репозиторий
3. Render использует `render.yaml` из репозитория
4. В настройках Environment добавь `BOT_TOKEN` и ключи нужных провайдеров (`GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `KIE_API_KEY` или `OPENAI_API_KEY`), затем нажми **Deploy latest commit**. Для голосовых и фотографий дополнительно можно задать публичные ссылки `TELEGRAM_VOICE_URL` и `TELEGRAM_PHOTO_URL`.

Telegram подключается автоматически через long polling после запуска Render. Отдельный webhook и отдельный сервис не нужны.

Сервис обязан запускаться командой `npm start` и слушать порт, который Render передает в переменной `PORT`.

### Настройка UptimeRobot

Для проверки доступности создай HTTP(s)-монитор с URL:

```text
https://<имя-сервиса>.onrender.com/api/health
```

Ожидаемый ответ: JSON со статусом `ok`. Пинговать локальный `localhost` или URL из dev container бесполезно: UptimeRobot должен обращаться к публичному адресу Render.

После деплоя существующий монитор UptimeRobot продолжит проверять `/api/health`; менять его не нужно.

На бесплатном тарифе Render может засыпать после периода бездействия. Внешний мониторинг иногда уменьшает такие простои, но не является гарантией работы 24/7; для постоянной работы нужен платный план.

Ключи добавляй только в Secrets Render или локальный `.env`. Файл `ключ` уже исключён из Git. Так как ключи были опубликованы в чате, их следует отозвать и выпустить заново перед деплоем.

### Почему локальные программы не помогали

Скрипт, запущенный на компьютере, не удерживает процесс Render и не заменяет внешний HTTP-монитор. Кроме того, до исправления порта запросы Render попадали на порт `PORT`, а приложение слушало только `3000`, поэтому health-check не проходил.

Для ручной проверки:

```bash
curl https://<имя-сервиса>.onrender.com/api/health
```

### Чтобы сервис не засыпал на бесплатном тарифе:
1. Зарегистрируйся на [UptimeRobot](https://uptimerobot.com)
2. Добавь монитор: HTTP(s) → укажи URL `/api/health`
3. Проверь, что монитор получает HTTP `200`

## Файлы

| Файл | Описание |
|------|----------|
| `server.ts` | Express API и production-сервер приложения |
| `src/` | React-интерфейс и 3D просмотрщик |
| `hori_memory.json` | Память о пользователе (факты, диалоги) |
| `hori_personality.json` | Личность Хори (саморазвивающаяся) |
| `hori_diary.json` | Личный дневник Хори |
| `hori_knowledge.json` | Факты о Хори и мире Horimiya |
| `hori_photos/` | Оригинальные изображения Хори по настроению |
| `render.yaml` | Конфиг деплоя на Render |


## Архитектура удалённого мозга Xori

Xori теперь умеет работать по схеме **Render bridge → Hugging Face Space**. Render принимает Telegram/API-запрос, но не обязан загружать тяжёлые веса разговорной модели. Если задан `XORI_HF_SPACE_URL`, удалённая модель становится первым маршрутом генерации; локальные модели остаются резервом.

В репозитории подготовлен шаблон Space в `hf_space/`:
- `hf_space/app.py` — Gradio API с endpoint `generate`;
- `hf_space/requirements.txt` — зависимости;
- `hf_space/README.md` — конфигурация Space.

Hugging Face Spaces поддерживает именованные Gradio API endpoints, а API можно вызывать через `/gradio_api/call/<endpoint>`. urlДокументация Hugging Face Spaces APIhttps://huggingface.co/docs/hub/spaces-overview

Для бесплатного ZeroGPU есть ограничения по квоте и условиям аккаунта, поэтому ZeroGPU не следует считать безлимитным постоянным GPU. urlДокументация Hugging Face ZeroGPUhttps://huggingface.co/docs/hub/spaces-zerogpu

### Переменные Render

Установи:
```text
XORI_HF_SPACE_URL=https://<namespace>-<space>.hf.space
XORI_HF_ENDPOINT=generate
XORI_HF_MODEL_ID=<namespace>/<model-repo>
XORI_HF_BRIDGE_TOKEN=<случайный-секрет>
```

Тот же `XORI_HF_BRIDGE_TOKEN` должен использоваться клиентом, который вызывает `/api/xori`. Сам токен не добавляй в Git.

OpenAPI-схема для клиента находится в `docs/xori-openapi.yaml`.

### Хранение модели

Если модель нужно автоматически отправлять из GitHub Actions в Hugging Face Hub, используй GitHub Secret `HF_TOKEN` с правом записи и репозиторий модели вроде `Abobus2222228/Xoritg`. Hugging Face официально поддерживает `create_repo` и `upload_folder` для такой схемы. 
