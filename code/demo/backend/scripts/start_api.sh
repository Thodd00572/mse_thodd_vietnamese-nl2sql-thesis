#!/bin/bash

# Vietnamese NL2SQL API Server Startup Script

echo "Starting Vietnamese NL2SQL API Server..."

# Check if model path is provided
if [ -z "$1" ]; then
    echo "ERROR: Model path required"
    echo "Usage: ./start_api.sh <model_path>"
    echo "Example: ./start_api.sh /path/to/phobert_sql_trained"
    exit 1
fi

MODEL_PATH="$1"
HOST="${2:-0.0.0.0}"
PORT="${3:-8000}"

echo "Model path: $MODEL_PATH"
echo "Host: $HOST"
echo "Port: $PORT"

# Check if model files exist
if [ ! -f "${MODEL_PATH}.pth" ]; then
    echo "ERROR: Model file not found: ${MODEL_PATH}.pth"
    exit 1
fi

if [ ! -d "${MODEL_PATH}_tokenizer" ]; then
    echo "WARNING: Tokenizer directory not found: ${MODEL_PATH}_tokenizer"
    echo "Will use default PhoBERT tokenizer"
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting API server..."
python api_server.py \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT"
