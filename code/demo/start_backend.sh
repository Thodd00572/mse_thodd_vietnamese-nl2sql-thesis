#!/bin/bash

# Vietnamese NL2SQL Backend Startup Script
# This script starts the FastAPI backend server on port 8000

echo "🚀 Starting Vietnamese NL2SQL Backend Server..."
echo "─────────────────────────────────────────────────"

# Navigate to backend core directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/backend/core"

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8000 is already in use!"
    echo "Finding process..."
    PID=$(lsof -Pi :8000 -sTCP:LISTEN -t)
    echo "Process PID: $PID"
    read -p "Kill this process and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Killing process $PID..."
        kill -9 $PID
        sleep 2
    else
        echo "Exiting. Please stop the existing process first."
        exit 1
    fi
fi

# Start the server
echo "Starting backend server on http://localhost:8000..."
echo "Press CTRL+C to stop the server"
echo "─────────────────────────────────────────────────"
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
