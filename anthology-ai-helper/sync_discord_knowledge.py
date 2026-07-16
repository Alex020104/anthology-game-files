from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import discord


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "discord_config.json"


@dataclass
class Config:
    approved_reactions: set[str]
    channels: dict[str, int]
    max_messages_per_channel: int
    output_dir: Path


def load_config() -> Config:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"Missing {CONFIG_PATH}. Copy discord_config.example.json to discord_config.json and fill channel IDs."
        )
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    channels = {
        str(name): int(channel_id)
        for name, channel_id in raw.get("channels", {}).items()
        if str(channel_id).strip() and "PUT_" not in str(channel_id)
    }
    if not channels:
        raise SystemExit("discord_config.json has no configured channels.")
    return Config(
        approved_reactions={str(item) for item in raw.get("approved_reactions", ["✅", "📌"])},
        channels=channels,
        max_messages_per_channel=int(raw.get("max_messages_per_channel", 200)),
        output_dir=(ROOT / raw.get("output_dir", "knowledge/discord")).resolve(),
    )


def clean_text(value: str) -> str:
    value = re.sub(r"<@!?\d+>", "@user", value or "")
    value = re.sub(r"<#\d+>", "#channel", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def has_approved_reaction(message: discord.Message, approved: set[str]) -> bool:
    for reaction in message.reactions:
        if str(reaction.emoji) in approved and reaction.count > 0:
            return True
    return False


async def sync() -> None:
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set DISCORD_BOT_TOKEN first.")

    config = load_config()
    intents = discord.Intents.default()
    intents.guilds = True
    intents.messages = True
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready() -> None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        index: list[str] = []
        try:
            for name, channel_id in config.channels.items():
                channel = client.get_channel(channel_id) or await client.fetch_channel(channel_id)
                if not isinstance(channel, (discord.TextChannel, discord.Thread)):
                    print(f"Skip {name}: unsupported channel type {type(channel).__name__}")
                    continue
                entries: list[str] = []
                async for message in channel.history(limit=config.max_messages_per_channel):
                    if message.author.bot:
                        continue
                    if not has_approved_reaction(message, config.approved_reactions):
                        continue
                    text = clean_text(message.content)
                    if not text:
                        continue
                    attachments = "\n".join(a.url for a in message.attachments)
                    block = [
                        f"### {message.created_at:%Y-%m-%d} — {message.author.display_name}",
                        text,
                        f"Source: {message.jump_url}",
                    ]
                    if attachments:
                        block.append("Attachments:\n" + attachments)
                    entries.append("\n\n".join(block))
                output = config.output_dir / f"{name}.md"
                output.write_text(
                    "# Discord knowledge: " + name + "\n\n" + "\n\n---\n\n".join(reversed(entries)) + "\n",
                    encoding="utf-8",
                )
                index.append(f"- {name}: {len(entries)} approved messages → {output.relative_to(ROOT)}")
                print(index[-1])
        finally:
            (config.output_dir / "index.md").write_text("# Discord sync index\n\n" + "\n".join(index) + "\n", encoding="utf-8")
            await client.close()

    await client.start(token)


if __name__ == "__main__":
    asyncio.run(sync())

