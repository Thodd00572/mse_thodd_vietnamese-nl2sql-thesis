#!/bin/bash

# Vietnamese-to-SQL Translation Thesis Experiment Startup Script

echo "Starting MSE Thesis Experimental App..."
echo "================================================"

# Check if we're in the right directory
if [ ! -f "backend/requirements.txt" ]; then
    echo "ERROR: Please run this script from the code/ directory"
    exit 1
fi

# Create Python virtual environment for backend
echo "Setting up Python environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Start backend server in background
echo "Starting FastAPI backend server..."
python start_server.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 5

# Setup frontend
echo "Setting up React/Next.js frontend..."
cd ../frontend

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

# Start frontend development server
echo "Starting Next.js frontend server..."
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "Thesis experimental app is starting up!"
echo "================================================"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Available Pages:"
echo "   • Search: http://localhost:3000/"
echo "   • Analysis: http://localhost:3000/analysis"
echo "   • Database: http://localhost:3000/database"
echo ""
echo "To stop the servers:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""

# Keep script running
wait
