#!/usr/bin/env python3
"""
Startup script for the Vietnamese-to-SQL Translation Thesis System
Starts both Backend (FastAPI) and Frontend (Next.js) servers
"""

import sys
import os
import subprocess
import threading
import time
import logging
import signal
import webbrowser
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global process tracking
processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info(" Shutting down servers...")
    for process in processes:
        if process.poll() is None:  # Process is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    sys.exit(0)

def start_backend():
    """Start the FastAPI backend server"""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Starting Backend at {backend_dir}")
    
    try:
        # Set environment variable to skip local models
        env = os.environ.copy()
        env['SKIP_LOCAL_MODELS'] = 'true'
        
        process = subprocess.Popen([
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, '{backend_dir}')
import uvicorn
uvicorn.run(
    'main:app',
    host='0.0.0.0',
    port=8000,
    reload=True,
    reload_dirs=['{backend_dir}'],
    log_level='info'
)
"""
        ], cwd=backend_dir, env=env)
        processes.append(process)
        logger.info("Backend server started on http://localhost:8000")
        return process
    except Exception as e:
        logger.error(f"Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the Next.js frontend server"""
    project_root = Path(__file__).parent.parent
    frontend_dir = project_root / "frontend"
    
    if not frontend_dir.exists():
        logger.error(f"Frontend directory not found: {frontend_dir}")
        return None
    
    logger.info(f"Starting Frontend at {frontend_dir}")
    
    try:
        # Check if node_modules exists, if not run npm install
        if not (frontend_dir / "node_modules").exists():
            logger.info(" Installing frontend dependencies...")
            install_process = subprocess.run([
                "npm", "install"
            ], cwd=frontend_dir, capture_output=True, text=True)
            
            if install_process.returncode != 0:
                logger.error(f"npm install failed: {install_process.stderr}")
                return None
            logger.info("Frontend dependencies installed")
        
        # Start the development server
        process = subprocess.Popen([
            "npm", "run", "dev"
        ], cwd=frontend_dir)
        processes.append(process)
        logger.info("Frontend server started on http://localhost:3000")
        return process
    except Exception as e:
        logger.error(f"Failed to start frontend: {e}")
        return None

def wait_for_servers():
    """Wait for servers to be ready and open browser"""
    import requests
    
    # Wait for backend
    backend_ready = False
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8000/docs", timeout=2)
            if response.status_code == 200:
                backend_ready = True
                logger.info("Backend is ready")
                break
        except:
            pass
        time.sleep(1)
    
    # Wait for frontend
    frontend_ready = False
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:3000", timeout=2)
            if response.status_code == 200:
                frontend_ready = True
                logger.info("Frontend is ready")
                break
        except:
            pass
        time.sleep(1)
    
    if backend_ready and frontend_ready:
        logger.info("Opening application in browser...")
        webbrowser.open("http://localhost:3000")
    elif not backend_ready:
        logger.warning("Backend may not be ready yet")
    elif not frontend_ready:
        logger.warning("Frontend may not be ready yet")

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Starting Vietnamese-to-SQL Translation Thesis System...")
    print("Backend: FastAPI server with Colab-only models")
    print("Frontend: Next.js React application")
    print("Press Ctrl+C to stop all servers")
    print("-" * 60)
    
    # Start backend server
    backend_process = start_backend()
    if not backend_process:
        logger.error("Failed to start backend server")
        sys.exit(1)
    
    # Wait a moment for backend to initialize
    time.sleep(3)
    
    # Start frontend server
    frontend_process = start_frontend()
    if not frontend_process:
        logger.error("Failed to start frontend server")
        # Kill backend if frontend fails
        if backend_process.poll() is None:
            backend_process.terminate()
        sys.exit(1)
    
    # Wait for servers to be ready and open browser
    threading.Thread(target=wait_for_servers, daemon=True).start()
    
    print("\n" + "=" * 60)
    print("SYSTEM READY!")
    print("Backend API: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Frontend App: http://localhost:3000")
    print("=" * 60)
    print("The application will open automatically in your browser")
    print("Press Ctrl+C to stop all servers")
    
    try:
        # Keep the main thread alive
        while True:
            # Check if processes are still running
            if backend_process.poll() is not None:
                logger.error("Backend process died")
                break
            if frontend_process.poll() is not None:
                logger.error("Frontend process died")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
