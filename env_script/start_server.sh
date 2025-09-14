#!/bin/bash

# MSE Thesis 2025 - Vietnamese NL2SQL Server Startup Script
# This script starts both the backend and frontend servers

echo "🚀 Starting MSE Thesis 2025 Vietnamese NL2SQL System..."
echo "=================================================="

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/code/backend"
FRONTEND_DIR="$PROJECT_ROOT/code/frontend"

echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to start backend server
start_backend() {
    echo "🔧 Starting Backend Server (Port 8000)..."
    
    if check_port 8000; then
        echo "⚠️  Port 8000 is already in use. Backend may already be running."
        echo "   Visit: http://localhost:8000"
    else
        cd "$BACKEND_DIR"
        echo "   Directory: $BACKEND_DIR"
        echo "   Starting uvicorn server..."
        
        # Start backend in background
        python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
        BACKEND_PID=$!
        
        # Wait a moment for server to start
        sleep 3
        
        if check_port 8000; then
            echo "✅ Backend server started successfully!"
            echo "   URL: http://localhost:8000"
            echo "   PID: $BACKEND_PID"
            echo "   Logs: $BACKEND_DIR/backend.log"
        else
            echo "❌ Failed to start backend server. Check logs at $BACKEND_DIR/backend.log"
        fi
    fi
    echo ""
}

# Function to start frontend server
start_frontend() {
    echo "🎨 Starting Frontend Server (Port 3000)..."
    
    if check_port 3000; then
        echo "⚠️  Port 3000 is already in use. Frontend may already be running."
        echo "   Visit: http://localhost:3000"
    else
        cd "$FRONTEND_DIR"
        echo "   Directory: $FRONTEND_DIR"
        
        # Check if build exists, if not build first
        if [ ! -d ".next" ]; then
            echo "   Building Next.js application..."
            npm run build
        fi
        
        echo "   Starting Next.js server..."
        
        # Start frontend in background
        npm start > frontend.log 2>&1 &
        FRONTEND_PID=$!
        
        # Wait a moment for server to start
        sleep 3
        
        if check_port 3000; then
            echo "✅ Frontend server started successfully!"
            echo "   URL: http://localhost:3000"
            echo "   PID: $FRONTEND_PID"
            echo "   Logs: $FRONTEND_DIR/frontend.log"
        else
            echo "❌ Failed to start frontend server. Check logs at $FRONTEND_DIR/frontend.log"
        fi
    fi
    echo ""
}

# Function to display final status
show_status() {
    echo "📊 Server Status Summary:"
    echo "========================"
    
    if check_port 8000; then
        echo "✅ Backend:  http://localhost:8000 (Running)"
    else
        echo "❌ Backend:  http://localhost:8000 (Not Running)"
    fi
    
    if check_port 3000; then
        echo "✅ Frontend: http://localhost:3000 (Running)"
    else
        echo "❌ Frontend: http://localhost:3000 (Not Running)"
    fi
    
    echo ""
    echo "🔗 Quick Links:"
    echo "   • Frontend UI: http://localhost:3000"
    echo "   • Backend API: http://localhost:8000"
    echo "   • API Docs: http://localhost:8000/docs"
    echo ""
    echo "📝 Features Available:"
    echo "   • Pipeline 1: Vietnamese → PhoBERT-SQL → SQL"
    echo "   • Pipeline 2: Vietnamese → English → SQL"
    echo "   • Google Colab Integration"
    echo "   • Performance Analytics"
    echo ""
    echo "🛑 To stop servers: Use Ctrl+C or run 'pkill -f uvicorn' and 'pkill -f next'"
}

# Main execution
echo "🔍 Checking system requirements..."

# Check if required directories exist
if [ ! -d "$BACKEND_DIR" ]; then
    echo "❌ Backend directory not found: $BACKEND_DIR"
    exit 1
fi

if [ ! -d "$FRONTEND_DIR" ]; then
    echo "❌ Frontend directory not found: $FRONTEND_DIR"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if Node.js/npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ Node.js/npm is required but not installed"
    exit 1
fi

echo "✅ System requirements check passed"
echo ""

# Start servers
start_backend
start_frontend
show_status

echo "🎉 Vietnamese NL2SQL System is ready!"
echo "   Open http://localhost:3000 in your browser to get started."
