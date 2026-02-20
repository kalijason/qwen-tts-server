#!/usr/bin/env python3
"""
Qwen TTS Server - HTTP server for Qwen3-TTS text-to-speech
Drop-in replacement for mac-tts with enhanced AI voice synthesis

Endpoints:
- POST /say - Generate and play TTS (compatible with mac-tts)
- POST /generate - Generate audio file without playing
- GET /health - Health check
- GET /voices - List available speakers
- GET /status - Model status
- POST /preload - Preload model into memory

Cache Management:
- GET /cache/stats - Cache statistics (hits, misses, size)
- GET /cache/list - List all cached entries
- POST /cache/clear - Clear all cache
- DELETE /cache/delete/<hash> - Delete specific cache entry

Environment Variables:
- QWEN_TTS_CACHE: Cache directory (default: /tmp/qwen-tts-cache)
- QWEN_TTS_CACHE_MAX_MB: Max cache size in MB (default: 500)
"""

import os
import sys
import argparse
import tempfile
import subprocess
import hashlib
import time
import threading
import json
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify, send_file
from collections import OrderedDict

# Lazy imports for heavy dependencies
_model = None
_model_lock = threading.Lock()

app = Flask(__name__)

# Configuration
DEFAULT_PORT = int(os.environ.get("QWEN_TTS_PORT", "5051"))
DEFAULT_SPEAKER = os.environ.get("QWEN_TTS_SPEAKER", "serena")
DEFAULT_LANGUAGE = os.environ.get("QWEN_TTS_LANGUAGE", "Chinese")
MODEL_PATH = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
CACHE_DIR = Path(os.environ.get("QWEN_TTS_CACHE", "/tmp/qwen-tts-cache"))
CACHE_MAX_SIZE_MB = int(os.environ.get("QWEN_TTS_CACHE_MAX_MB", "500"))  # 500MB default
CACHE_INDEX_FILE = CACHE_DIR / ".cache_index.json"

# Cache statistics
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "total_requests": 0,
}
_cache_stats_lock = threading.Lock()

# LRU cache index: {hash: {"path": str, "size": int, "last_access": float, "text_preview": str}}
_cache_index: OrderedDict = OrderedDict()
_cache_index_lock = threading.Lock()


def _load_cache_index():
    """Load cache index from disk"""
    global _cache_index
    if CACHE_INDEX_FILE.exists():
        try:
            with open(CACHE_INDEX_FILE, "r") as f:
                data = json.load(f)
                _cache_index = OrderedDict(data)
                print(f"📂 Loaded cache index: {len(_cache_index)} entries")
        except Exception as e:
            print(f"⚠️ Failed to load cache index: {e}")
            _cache_index = OrderedDict()
    # Also scan for orphaned cache files
    _rebuild_cache_index()


def _rebuild_cache_index():
    """Rebuild cache index from existing files"""
    global _cache_index
    if not CACHE_DIR.exists():
        return
    
    with _cache_index_lock:
        existing_hashes = set(_cache_index.keys())
        for wav_file in CACHE_DIR.glob("*.wav"):
            hash_key = wav_file.stem
            if hash_key not in existing_hashes:
                stat = wav_file.stat()
                _cache_index[hash_key] = {
                    "path": str(wav_file),
                    "size": stat.st_size,
                    "last_access": stat.st_mtime,
                    "text_preview": "(recovered)",
                }
        _save_cache_index_unlocked()


def _save_cache_index_unlocked():
    """Save cache index to disk (caller must hold lock)"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_INDEX_FILE, "w") as f:
            json.dump(dict(_cache_index), f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save cache index: {e}")


def _save_cache_index():
    """Save cache index to disk"""
    with _cache_index_lock:
        _save_cache_index_unlocked()


def _get_cache_size_bytes() -> int:
    """Get total cache size in bytes"""
    with _cache_index_lock:
        return sum(entry.get("size", 0) for entry in _cache_index.values())


def _evict_lru_if_needed(new_file_size: int = 0):
    """Evict least recently used entries if cache exceeds max size"""
    max_bytes = CACHE_MAX_SIZE_MB * 1024 * 1024
    
    with _cache_index_lock:
        current_size = sum(entry.get("size", 0) for entry in _cache_index.values())
        
        while current_size + new_file_size > max_bytes and _cache_index:
            # Remove oldest (first) entry
            oldest_hash, oldest_entry = _cache_index.popitem(last=False)
            old_path = Path(oldest_entry["path"])
            if old_path.exists():
                try:
                    old_path.unlink()
                    print(f"🗑️ Cache evicted: {oldest_hash[:8]}... ({oldest_entry.get('text_preview', '')[:20]})")
                except Exception as e:
                    print(f"⚠️ Failed to delete cache file: {e}")
            current_size -= oldest_entry.get("size", 0)
        
        _save_cache_index_unlocked()

# Available speakers (Qwen3-TTS CustomVoice preset voices)
# Reference: https://huggingface.co/Qwen/Qwen3-TTS
SPEAKERS = {
    # Female voices
    "serena": {"gender": "female", "style": "warm, friendly", "lang": "zh/en"},
    "vivian": {"gender": "female", "style": "professional", "lang": "zh/en"},
    "sohee": {"gender": "female", "style": "bright, youthful", "lang": "ko/en"},
    # Male voices
    "aiden": {"gender": "male", "style": "calm, mature", "lang": "en"},
    "dylan": {"gender": "male", "style": "energetic", "lang": "en"},
    "eric": {"gender": "male", "style": "warm, conversational", "lang": "en"},
    "ryan": {"gender": "male", "style": "professional", "lang": "en"},
    "uncle_fu": {"gender": "male", "style": "storytelling, elderly", "lang": "zh"},
    # Special
    "ono_anna": {"gender": "female", "style": "anime, cute", "lang": "ja"},
}


def get_model():
    """Lazy load the Qwen3-TTS model"""
    global _model
    
    if _model is not None:
        return _model
    
    with _model_lock:
        if _model is not None:
            return _model
        
        print(f"🔄 Loading Qwen3-TTS model: {MODEL_PATH}")
        start_time = time.time()
        
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            
            # Detect device - MPS has numerical issues with TTS, use CPU for stability
            if torch.cuda.is_available():
                device = "cuda:0"
                dtype = torch.bfloat16
            else:
                # CPU is more stable for Apple Silicon TTS
                device = "cpu"
                dtype = torch.float32
            
            print(f"   Using device: {device}")
            
            _model = Qwen3TTSModel.from_pretrained(
                MODEL_PATH,
                dtype=dtype,
                attn_implementation="sdpa",  # Compatible with MPS
                device_map=device
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Model loaded in {elapsed:.1f}s")
            
            return _model
            
        except ImportError as e:
            print(f"❌ Failed to import Qwen3-TTS: {e}")
            print("   Run: pip install qwen-tts torch soundfile")
            raise
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise


def get_cache_key(text: str, speaker: str, language: str, instruct: str = "") -> str:
    """Generate cache key based on content hash"""
    content = f"{text}|{speaker}|{language}|{instruct}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_cache_path(text: str, speaker: str, language: str, instruct: str = "") -> Path:
    """Generate cache file path based on content hash"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hash_key = get_cache_key(text, speaker, language, instruct)
    return CACHE_DIR / f"{hash_key}.wav"


def generate_speech(
    text: str,
    speaker: str = DEFAULT_SPEAKER,
    language: str = DEFAULT_LANGUAGE,
    instruct: str = "",
    use_cache: bool = True
) -> tuple[Path, bool]:
    """Generate speech and return the audio file path and cache hit status"""
    import soundfile as sf
    
    hash_key = get_cache_key(text, speaker, language, instruct)
    cache_path = CACHE_DIR / f"{hash_key}.wav"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Update stats
    with _cache_stats_lock:
        _cache_stats["total_requests"] += 1
    
    # Check cache
    if use_cache and cache_path.exists():
        with _cache_stats_lock:
            _cache_stats["hits"] += 1
        
        # Update LRU access time
        with _cache_index_lock:
            if hash_key in _cache_index:
                # Move to end (most recently used)
                _cache_index.move_to_end(hash_key)
                _cache_index[hash_key]["last_access"] = time.time()
        
        print(f"✅ Cache HIT: {hash_key[:8]}... ({text[:30]}{'...' if len(text) > 30 else ''})")
        return cache_path, True
    
    # Cache miss - generate
    with _cache_stats_lock:
        _cache_stats["misses"] += 1
    
    print(f"⏳ Cache MISS: {hash_key[:8]}... generating...")
    
    model = get_model()
    
    # Generate speech using CustomVoice API
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct if instruct else None,
    )
    
    # Save to cache
    sf.write(str(cache_path), wavs[0], sr)
    file_size = cache_path.stat().st_size
    
    # Evict old entries if needed
    _evict_lru_if_needed(file_size)
    
    # Add to index
    with _cache_index_lock:
        _cache_index[hash_key] = {
            "path": str(cache_path),
            "size": file_size,
            "last_access": time.time(),
            "text_preview": text[:50],
            "speaker": speaker,
            "language": language,
        }
        _cache_index.move_to_end(hash_key)  # Mark as most recently used
        _save_cache_index_unlocked()
    
    print(f"💾 Cached: {hash_key[:8]}... ({file_size / 1024:.1f} KB)")
    
    return cache_path, False


def play_audio(audio_path: Path):
    """Play audio file using system command"""
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(audio_path)], check=True)
    elif sys.platform == "linux":
        subprocess.run(["aplay", str(audio_path)], check=True)
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


@app.route("/say", methods=["POST"])
def say():
    """
    Trigger TTS notification (compatible with mac-tts)
    
    POST /say {
        "message": "Hello",
        "voice": "Chelsie",      # Speaker name
        "instruct": "用開心的語氣說",  # Optional: emotion/style instruction
        "async": false           # Optional: return immediately
    }
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    speaker = data.get("voice", data.get("speaker", DEFAULT_SPEAKER))
    language = data.get("language", DEFAULT_LANGUAGE)
    instruct = data.get("instruct", "")
    async_mode = data.get("async", False)
    
    if not message:
        return jsonify({"error": 'Missing "message" parameter'}), 400
    
    try:
        if async_mode:
            # Generate and play in background
            def bg_task():
                audio_path, _ = generate_speech(message, speaker, language, instruct)
                play_audio(audio_path)
            
            thread = threading.Thread(target=bg_task, daemon=True)
            thread.start()
            return jsonify({"success": True, "message": message, "async": True})
        
        # Synchronous mode
        start_time = time.time()
        audio_path, cache_hit = generate_speech(message, speaker, language, instruct)
        gen_time = time.time() - start_time
        
        play_audio(audio_path)
        total_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "message": message,
            "speaker": speaker,
            "language": language,
            "instruct": instruct if instruct else None,
            "cache_hit": cache_hit,
            "generation_time": round(gen_time, 2),
            "total_time": round(total_time, 2),
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate audio file without playing
    
    POST /generate {
        "message": "Hello",
        "voice": "Chelsie",
        "instruct": "用開心的語氣說",
        "format": "wav"  # wav or mp3
    }
    
    Returns: Audio file
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    speaker = data.get("voice", data.get("speaker", DEFAULT_SPEAKER))
    language = data.get("language", DEFAULT_LANGUAGE)
    instruct = data.get("instruct", "")
    fmt = data.get("format", "wav")
    
    if not message:
        return jsonify({"error": 'Missing "message" parameter'}), 400
    
    try:
        audio_path, cache_hit = generate_speech(message, speaker, language, instruct)
        
        # Convert to MP3 if requested
        if fmt == "mp3":
            mp3_path = audio_path.with_suffix(".mp3")
            if not mp3_path.exists():
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(audio_path),
                    "-codec:a", "libmp3lame", "-qscale:a", "2",
                    str(mp3_path)
                ], check=True, capture_output=True)
            response = send_file(mp3_path, mimetype="audio/mpeg")
            response.headers["X-Cache-Hit"] = str(cache_hit).lower()
            return response
        
        response = send_file(audio_path, mimetype="audio/wav")
        response.headers["X-Cache-Hit"] = str(cache_hit).lower()
        return response
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint (compatible with mac-tts)"""
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/voices", methods=["GET"])
def voices():
    """List available speakers"""
    return jsonify({
        "speakers": SPEAKERS,
        "default": DEFAULT_SPEAKER,
        "note": "Use 'instruct' parameter for emotion/style control"
    })


@app.route("/status", methods=["GET"])
def status():
    """Model and server status"""
    import torch
    
    model_loaded = _model is not None
    
    return jsonify({
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "cache_dir": str(CACHE_DIR),
        "default_speaker": DEFAULT_SPEAKER,
        "default_language": DEFAULT_LANGUAGE,
    })


@app.route("/preload", methods=["POST"])
def preload():
    """Preload the model into memory"""
    try:
        get_model()
        return jsonify({"success": True, "message": "Model loaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/cache/stats", methods=["GET"])
def cache_stats():
    """Get cache statistics"""
    with _cache_stats_lock:
        stats = _cache_stats.copy()
    
    total = stats["total_requests"]
    hit_rate = (stats["hits"] / total * 100) if total > 0 else 0
    
    cache_size = _get_cache_size_bytes()
    
    with _cache_index_lock:
        entry_count = len(_cache_index)
    
    return jsonify({
        "hits": stats["hits"],
        "misses": stats["misses"],
        "total_requests": total,
        "hit_rate_percent": round(hit_rate, 1),
        "cache_entries": entry_count,
        "cache_size_mb": round(cache_size / 1024 / 1024, 2),
        "cache_max_size_mb": CACHE_MAX_SIZE_MB,
        "cache_dir": str(CACHE_DIR),
    })


@app.route("/cache/list", methods=["GET"])
def cache_list():
    """List cached entries"""
    with _cache_index_lock:
        entries = []
        for hash_key, entry in _cache_index.items():
            entries.append({
                "hash": hash_key,
                "text_preview": entry.get("text_preview", ""),
                "speaker": entry.get("speaker", ""),
                "size_kb": round(entry.get("size", 0) / 1024, 1),
                "last_access": entry.get("last_access", 0),
            })
    
    # Sort by last access (most recent first)
    entries.sort(key=lambda x: x["last_access"], reverse=True)
    
    return jsonify({
        "entries": entries,
        "count": len(entries),
    })


@app.route("/cache/clear", methods=["POST"])
def cache_clear():
    """Clear all cache entries"""
    global _cache_index
    
    cleared_count = 0
    cleared_size = 0
    
    with _cache_index_lock:
        for hash_key, entry in list(_cache_index.items()):
            path = Path(entry["path"])
            if path.exists():
                try:
                    cleared_size += entry.get("size", 0)
                    path.unlink()
                    cleared_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete {path}: {e}")
        
        _cache_index = OrderedDict()
        _save_cache_index_unlocked()
    
    # Reset stats
    with _cache_stats_lock:
        _cache_stats["hits"] = 0
        _cache_stats["misses"] = 0
        _cache_stats["total_requests"] = 0
    
    print(f"🗑️ Cache cleared: {cleared_count} files, {cleared_size / 1024 / 1024:.2f} MB")
    
    return jsonify({
        "success": True,
        "cleared_entries": cleared_count,
        "cleared_size_mb": round(cleared_size / 1024 / 1024, 2),
    })


@app.route("/cache/delete/<hash_key>", methods=["DELETE"])
def cache_delete(hash_key: str):
    """Delete a specific cache entry"""
    with _cache_index_lock:
        if hash_key not in _cache_index:
            return jsonify({"error": "Cache entry not found"}), 404
        
        entry = _cache_index.pop(hash_key)
        path = Path(entry["path"])
        
        if path.exists():
            path.unlink()
        
        _save_cache_index_unlocked()
    
    return jsonify({
        "success": True,
        "deleted": hash_key,
        "text_preview": entry.get("text_preview", ""),
    })


def main():
    global DEFAULT_SPEAKER, DEFAULT_LANGUAGE, MODEL_PATH, CACHE_MAX_SIZE_MB
    
    parser = argparse.ArgumentParser(description="Qwen TTS HTTP API Server")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT,
                        help="Port to listen on")
    parser.add_argument("-v", "--voice", default=DEFAULT_SPEAKER,
                        help="Default speaker voice")
    parser.add_argument("-l", "--language", default=DEFAULT_LANGUAGE,
                        help="Default language (Chinese, English, etc.)")
    parser.add_argument("-m", "--model", default=MODEL_PATH,
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--preload", action="store_true",
                        help="Preload model on startup")
    parser.add_argument("--cache-max-mb", type=int, default=CACHE_MAX_SIZE_MB,
                        help="Maximum cache size in MB (default: 500)")
    args = parser.parse_args()
    
    DEFAULT_SPEAKER = args.voice
    DEFAULT_LANGUAGE = args.language
    MODEL_PATH = args.model
    CACHE_MAX_SIZE_MB = args.cache_max_mb
    
    # Initialize cache
    _load_cache_index()
    cache_size = _get_cache_size_bytes()
    
    print(f"🎙️ Qwen TTS Server starting on http://{args.host}:{args.port}")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Default speaker: {DEFAULT_SPEAKER}")
    print(f"   Default language: {DEFAULT_LANGUAGE}")
    print(f"   Cache: {CACHE_DIR} ({cache_size / 1024 / 1024:.1f} MB / {CACHE_MAX_SIZE_MB} MB max)")
    
    if args.preload:
        print("   Preloading model...")
        get_model()
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
