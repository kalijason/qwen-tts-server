# Qwen TTS Server

AI-powered text-to-speech HTTP server using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS), designed as a drop-in replacement for `mac-tts` with enhanced natural voice synthesis.

## Features

- 🎙️ **Qwen3-TTS**: State-of-the-art multilingual TTS with natural prosody
- 🎭 **Emotion Control**: Use natural language to control voice style
- 🔊 **Multiple Speakers**: 9 preset voices (Chinese, English, Japanese, Korean)
- ⚡ **Apple Silicon Optimized**: Native MPS/CPU support for M1/M2/M3/M4
- 🔌 **mac-tts Compatible**: Drop-in replacement API
- 💾 **Audio Caching**: LRU cache with configurable size limit

## Installation

### Quick Start (From Source)

```bash
git clone https://github.com/kalijason/qwen-tts-server
cd qwen-tts-server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python qwen_tts_server.py
```

### Via pip

```bash
pip install qwen-tts-server
qwen-tts
```

### Via Homebrew

```bash
brew tap kalijason/tap
brew install qwen-tts
brew services start qwen-tts
```

## Usage

### Start the Server

```bash
# Basic usage
python qwen_tts_server.py

# With options
python qwen_tts_server.py --port 5051 --voice serena --preload

# Environment variables
QWEN_TTS_PORT=5051 QWEN_TTS_SPEAKER=vivian python qwen_tts_server.py
```

### As a macOS Service

```bash
brew services start qwen-tts
brew services stop qwen-tts
brew services restart qwen-tts
```

## API Reference

### POST /say

Generate and play TTS audio (compatible with mac-tts).

```bash
curl -X POST http://localhost:5051/say \
  -H "Content-Type: application/json" \
  -d '{"message": "你好世界", "voice": "serena"}'

# With emotion control
curl -X POST http://localhost:5051/say \
  -H "Content-Type: application/json" \
  -d '{
    "message": "哇！外送到了！",
    "voice": "serena",
    "instruct": "用興奮開心的語氣說"
  }'

# Async mode (returns immediately)
curl -X POST http://localhost:5051/say \
  -d '{"message": "Hello", "voice": "aiden", "async": true}'
```

### POST /generate

Generate audio file without playing.

```bash
# Get WAV file
curl -X POST http://localhost:5051/generate \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello world", "voice": "serena"}' \
  -o output.wav

# Get MP3 file
curl -X POST http://localhost:5051/generate \
  -d '{"message": "Hello", "format": "mp3"}' \
  -o output.mp3
```

Response header `X-Cache-Hit` indicates if the audio was served from cache.

### GET /voices

List available speakers.

```bash
curl http://localhost:5051/voices
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:5051/health
```

### GET /status

Server and model status.

```bash
curl http://localhost:5051/status
```

### POST /preload

Preload model into memory (useful for faster first request).

```bash
curl -X POST http://localhost:5051/preload
```

## Cache Management

The server includes an LRU file cache to speed up repeated requests.

### GET /cache/stats

Get cache statistics.

```bash
curl http://localhost:5051/cache/stats
```

Response:
```json
{
  "hits": 42,
  "misses": 10,
  "hit_rate_percent": 80.8,
  "cache_entries": 52,
  "cache_size_mb": 15.3,
  "cache_max_size_mb": 500
}
```

### GET /cache/list

List all cached entries.

```bash
curl http://localhost:5051/cache/list
```

### POST /cache/clear

Clear all cache entries.

```bash
curl -X POST http://localhost:5051/cache/clear
```

### DELETE /cache/delete/:hash

Delete a specific cache entry.

```bash
curl -X DELETE http://localhost:5051/cache/delete/abc123
```

## Available Speakers

| Speaker | Gender | Style | Languages |
|---------|--------|-------|-----------|
| serena | Female | Warm, friendly | Chinese, English |
| vivian | Female | Professional | Chinese, English |
| sohee | Female | Bright, youthful | Korean, English |
| aiden | Male | Calm, mature | English |
| dylan | Male | Energetic | English |
| eric | Male | Warm, conversational | English |
| ryan | Male | Professional | English |
| uncle_fu | Male | Storytelling, elderly | Chinese |
| ono_anna | Female | Anime, cute | Japanese |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN_TTS_PORT` | 5051 | Server port |
| `QWEN_TTS_SPEAKER` | serena | Default speaker |
| `QWEN_TTS_LANGUAGE` | Chinese | Default language |
| `QWEN_TTS_MODEL` | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | Model path |
| `QWEN_TTS_CACHE` | /tmp/qwen-tts-cache | Cache directory |
| `QWEN_TTS_CACHE_MAX_MB` | 500 | Max cache size in MB |

## Home Assistant Integration

```yaml
# configuration.yaml
rest_command:
  qwen_tts:
    url: "http://localhost:5051/say"
    method: POST
    headers:
      Content-Type: application/json
    payload: '{"message": "{{ message }}", "voice": "serena"}'

# automation example
automation:
  - alias: "Doorbell notification"
    trigger:
      - platform: state
        entity_id: binary_sensor.doorbell
        to: "on"
    action:
      - service: rest_command.qwen_tts
        data:
          message: "有人按門鈴"
```

## Requirements

- Python 3.10+
- ~2GB disk space for model (downloaded on first run)
- ~4GB RAM recommended

## License

MIT
