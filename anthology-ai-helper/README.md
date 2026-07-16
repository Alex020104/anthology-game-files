# Anthology AI Helper

Локальный/серверный помощник для игроков Anthology.

## Быстрый тест

1. Установи ключ OpenAI только на машине, где запускается helper:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

2. Запусти сервер:

```powershell
py -3 server.py
```

3. В Chat Relay:

```text
/ai почему не обновляется Chernobyl Relay Chat?
```

## Переменные окружения

- `OPENAI_API_KEY` — ключ OpenAI, обязателен для настоящих ответов.
- `ANTHOLOGY_AI_MODEL` — модель, по умолчанию `gpt-4.1-mini`.
- `ANTHOLOGY_AI_HOST` — хост, по умолчанию `127.0.0.1`.
- `ANTHOLOGY_AI_PORT` — порт, по умолчанию `8787`.
- `ANTHOLOGY_AI_RATE_SECONDS` — лимит на клиента, по умолчанию `20`.

Если ключ не задан, сервер отвечает тестовым сообщением и не вызывает OpenAI.

## Полуавтоматическая база знаний из Discord

1. Создай Discord bot в Developer Portal.
2. Включи `MESSAGE CONTENT INTENT`.
3. Пригласи бота на сервер с правами читать сообщения.
4. Скопируй `discord_config.example.json` в `discord_config.json`.
5. Впиши ID каналов FAQ/гайдов/ошибок.
6. Помечай хорошие сообщения реакцией `✅` или `📌`.
7. Запусти:

```powershell
$env:DISCORD_BOT_TOKEN="discord_bot_token"
.\sync_discord_knowledge.ps1
```

Скрипт создаст файлы в `knowledge/discord/*.md`, и AI helper начнёт использовать их после перезапуска.

## Discord-бот для вопросов

Сначала запусти helper:

```powershell
.\start_helper.ps1
```

Потом бота:

```powershell
$env:DISCORD_BOT_TOKEN="discord_bot_token"
.\start_discord_bot.ps1
```

В Discord:

```text
!ai почему не обновляется чат?
```
