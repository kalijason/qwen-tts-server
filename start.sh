#!/bin/bash
# Start Qwen TTS Server
cd "$(dirname "$0")"
source .venv/bin/activate
exec python3 qwen_tts_server.py "$@"
