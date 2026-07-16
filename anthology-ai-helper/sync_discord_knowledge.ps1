$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
py -3 -m pip install -r requirements.txt
py -3 .\sync_discord_knowledge.py

