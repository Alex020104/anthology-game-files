from __future__ import annotations

import asyncio
import os
import urllib.error
import urllib.parse
import urllib.request

import discord


HELPER_URL = os.environ.get("ANTHOLOGY_AI_HELPER_URL", "http://127.0.0.1:8787/ask")
COMMAND_PREFIX = os.environ.get("ANTHOLOGY_AI_DISCORD_PREFIX", "!ai")
MAX_QUESTION_CHARS = 700


async def ask_helper(question: str) -> str:
    def request() -> str:
        data = question[:MAX_QUESTION_CHARS].encode("utf-8")
        req = urllib.request.Request(
            HELPER_URL,
            data=data,
            headers={"Content-Type": "text/plain; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.read().decode("utf-8", errors="replace") or f"AI helper error {exc.code}"
        except Exception as exc:
            return f"AI helper недоступен ({type(exc).__name__})."

    return await asyncio.to_thread(request)


class AnthologyAiBot(discord.Client):
    async def on_ready(self) -> None:
        print(f"Anthology AI Discord bot logged in as {self.user}")
        print(f"Command: {COMMAND_PREFIX} вопрос")

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        content = message.content.strip()
        if not content.lower().startswith(COMMAND_PREFIX.lower()):
            return
        question = content[len(COMMAND_PREFIX):].strip()
        if not question:
            await message.reply("Напиши вопрос после команды, например: `!ai почему не обновляется чат?`", mention_author=False)
            return
        async with message.channel.typing():
            answer = await ask_helper(question)
        await message.reply(answer[:1800], mention_author=False)


def main() -> None:
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set DISCORD_BOT_TOKEN first.")
    intents = discord.Intents.default()
    intents.guilds = True
    intents.messages = True
    intents.message_content = True
    AnthologyAiBot(intents=intents).run(token)


if __name__ == "__main__":
    main()

