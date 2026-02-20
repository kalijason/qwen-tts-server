# Homebrew formula for Qwen TTS Server
# 
# For local development:
#   brew tap kalijason/tap ~/path/to/homebrew-tap
#   brew install kalijason/tap/qwen-tts
#
# For public release:
#   1. Create GitHub release with tag (e.g., v1.0.0)
#   2. Update url and sha256 below
#   3. Push to homebrew-tap repo

class QwenTts < Formula
  desc "Qwen3-TTS HTTP API server for AI-powered text-to-speech"
  homepage "https://github.com/kalijason/qwen-tts-server"
  url "https://github.com/kalijason/qwen-tts-server/archive/refs/tags/v1.0.0.tar.gz"
  sha256 "REPLACE_WITH_ACTUAL_SHA256_AFTER_RELEASE"
  license "MIT"
  head "https://github.com/kalijason/qwen-tts-server.git", branch: "main"

  depends_on "python@3.12"
  depends_on "ffmpeg"

  def install
    # Create wrapper script that uses project's venv
    (bin/"qwen-tts").write <<~EOS
      #!/bin/bash
      cd "#{prefix}/libexec"
      source .venv/bin/activate 2>/dev/null || {
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -q flask torch soundfile qwen-tts numpy
      }
      exec python3 qwen_tts_server.py "$@"
    EOS
    chmod 0755, bin/"qwen-tts"

    # Install server script
    libexec.install "qwen_tts_server.py"
    libexec.install "requirements.txt"

    # Create log directory
    (var/"log").mkpath
  end

  def caveats
    <<~EOS
      Qwen TTS Server installed!

      First run will:
      - Create Python virtual environment
      - Install dependencies (~500MB)
      - Download Qwen3-TTS model (~1GB)

      Start manually:
        qwen-tts --port 5051

      Or as a service:
        brew services start qwen-tts

      Logs:
        tail -f #{var}/log/qwen-tts.log

      Available voices: serena, vivian, aiden, dylan, eric, ryan, sohee, uncle_fu, ono_anna

      API:
        curl -X POST http://localhost:5051/say \\
          -H "Content-Type: application/json" \\
          -d '{"message": "你好", "voice": "serena"}'
    EOS
  end

  service do
    run [opt_bin/"qwen-tts", "--port", "5051"]
    keep_alive true
    working_dir var
    log_path var/"log/qwen-tts.log"
    error_log_path var/"log/qwen-tts.error.log"
    environment_variables(
      QWEN_TTS_PORT: "5051",
      QWEN_TTS_SPEAKER: "serena",
      QWEN_TTS_CACHE: "/tmp/qwen-tts-cache",
      QWEN_TTS_CACHE_MAX_MB: "500"
    )
  end

  test do
    # Basic syntax check
    system "python3", "-c", "import ast; ast.parse(open('#{libexec}/qwen_tts_server.py').read())"
  end
end
