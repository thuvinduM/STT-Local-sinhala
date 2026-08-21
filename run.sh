#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run ./install.sh first."
    exit 1
fi

source venv/bin/activate

export LD_LIBRARY_PATH="$PWD/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

uvicorn app.main:app --host 0.0.0.0 --port 8007
