# Xori AI Studio

Отдельный control center для Xori и Training Engine.

## Сейчас
- Dashboard
- Chat UI
- Review ошибок: Исправить / Верно / Игнорировать
- Training Engine UI с автоматическим pipeline
- Разделы памяти, знаний, Telegram, системы и версий

Это первый рабочий UI-прототип. Пока Chat/Training/Test используют локальное состояние браузера.

## Backend-план
Studio должна обращаться к отдельному Control API, а не к Telegram bridge напрямую.

Основные API:
- GET /api/status
- POST /api/chat
- GET /api/errors
- POST /api/errors/:id/correct
- POST /api/errors/:id/true
- POST /api/errors/:id/ignore
- GET /api/training/jobs
- POST /api/training/start
- GET /api/training/jobs/:id
- POST /api/tests/run
- GET /api/versions
- POST /api/versions/:id/promote
- POST /api/versions/:id/rollback

Training Engine: подтверждённые исправления → очистка/dedup → JSONL → PEFT LoRA → candidate version → evaluation → сравнение → ручное подтверждение публикации.

Production LoRA не перезаписывается во время обучения.

## Deploy
Папка xori-studio предназначена для отдельного Render Static Site. Existing Telegram service остаётся отдельным.
