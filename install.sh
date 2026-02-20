#!/bin/bash
# Qwen TTS Server - Installation Script for macOS
set -e

echo "🎙️ Qwen TTS Server Installer"
echo "================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.10+ required. Found: $PYTHON_VERSION"
    echo "   Install with: brew install python@3.12"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment
VENV_DIR="$HOME/.qwen-tts"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate and install
source "$VENV_DIR/bin/activate"

echo "📥 Installing dependencies..."
pip install --upgrade pip wheel

# Install PyTorch with MPS support
echo "   Installing PyTorch (Apple Silicon optimized)..."
pip install torch torchvision torchaudio

# Install Qwen TTS
echo "   Installing Qwen TTS..."
pip install qwen-tts soundfile flask numpy

# Install this package
echo "   Installing qwen-tts-server..."
pip install -e "$(dirname "$0")"

# Create bin symlink
BIN_DIR="/usr/local/bin"
if [ ! -w "$BIN_DIR" ]; then
    BIN_DIR="$HOME/.local/bin"
    mkdir -p "$BIN_DIR"
fi

cat > "$BIN_DIR/qwen-tts" << 'EOF'
#!/bin/bash
source "$HOME/.qwen-tts/bin/activate"
exec python -m qwen_tts_server "$@"
EOF
chmod +x "$BIN_DIR/qwen-tts"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Usage:"
echo "  qwen-tts                    # Start server"
echo "  qwen-tts --preload          # Start with model preloaded"
echo "  qwen-tts --port 5051        # Custom port"
echo ""
echo "First run will download the model (~1GB)."
echo ""
echo "Test with:"
echo '  curl -X POST http://localhost:5051/say -d '\''{"message": "測試", "voice": "serena"}'\'''
