#!/usr/bin/env bash
# Sets up an isolated Python virtual environment for court-stt.
# Does NOT touch system Python, system CUDA, or the NVIDIA driver.
set -e

cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
    echo "python3 is required but was not found. Please install Python 3.10+."
    exit 1
fi

echo "==> Creating virtual environment (./venv)"
python3 -m venv venv
source venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing PyTorch (CPU build - only used for speaker embeddings, not for Whisper)"
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing project requirements"
pip install -r requirements.txt

echo ""
echo "==> Install complete."
echo "The Whisper large-v3 model (~3GB) and the SpeechBrain speaker"
echo "embedding model (~80MB, no HuggingFace token needed) will both"
echo "download automatically the first time you run the app."
echo "(requires internet access once)"
echo ""
echo "Run the app with: ./run.sh"
