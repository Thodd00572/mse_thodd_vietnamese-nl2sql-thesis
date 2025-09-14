#!/usr/bin/env python3
"""
Vietnamese NL2SQL System Startup Script
Coordinates PhoBERT training with API launch to ensure system integrity

Author: MSE14 Duong Dinh Tho
Thesis: Master of Software Engineering
© Copyright by MSE 14 Duong Dinh Tho, 2025
"""

import os
import sys
import time
import subprocess
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemCoordinator:
    """Coordinates PhoBERT training and API launch"""
    
    def __init__(self):
        self.backend_path = Path(__file__).parent / "backend"
        self.colab_notebook_path = Path(__file__).parent / "ColabNotebook" / "MSE_Thesis_Combined_Pipelines.py"
        self.api_process = None
        self.training_complete = False
        self.api_port = 8000
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        logger.info(" Checking system dependencies...")
        
        required_packages = [
            "fastapi", "uvicorn", "transformers", "torch", 
            "sqlite3", "pandas", "numpy", "requests"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                logger.info(f" {package}")
            except ImportError:
                missing_packages.append(package)
                logger.error(f" {package}")
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        logger.info(" All dependencies satisfied")
        return True
    
    def setup_database(self) -> bool:
        """Initialize the database"""
        logger.info(" Setting up database...")
        
        try:
            # Change to backend directory
            os.chdir(self.backend_path)
            
            # Import and initialize database
            sys.path.append(str(self.backend_path))
            from app import db_manager
            
            # Test database connection
            schema = db_manager.get_schema_info()
            product_count = schema.get("products", {}).get("row_count", 0)
            
            logger.info(f" Database ready with {product_count} products")
            return True
            
        except Exception as e:
            logger.error(f" Database setup failed: {e}")
            return False
    
    def check_training_status(self) -> Dict[str, Any]:
        """Check if PhoBERT training is complete"""
        try:
            response = requests.get(f"http://localhost:{self.api_port}/api/training/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        return {
            "training_status": {
                "is_training": False,
                "training_complete": False,
                "model_ready": False
            },
            "can_use_pipelines": False
        }
    
    def start_api_server(self) -> bool:
        """Start the FastAPI server"""
        logger.info(" Starting API server...")
        
        try:
            os.chdir(self.backend_path)
            
            # Start uvicorn server
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "app:app", 
                "--host", "0.0.0.0", 
                "--port", str(self.api_port),
                "--reload"
            ]
            
            self.api_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            for attempt in range(30):
                try:
                    response = requests.get(f"http://localhost:{self.api_port}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info(f" API server running on http://localhost:{self.api_port}")
                        return True
                except:
                    time.sleep(1)
            
            logger.error(" API server failed to start within 30 seconds")
            return False
            
        except Exception as e:
            logger.error(f" Failed to start API server: {e}")
            return False
    
    def display_colab_instructions(self):
        """Display instructions for Colab training"""
        logger.info("\n" + "="*80)
        logger.info(" PHOBERT TRAINING INSTRUCTIONS")
        logger.info("="*80)
        logger.info("1. Open Google Colab: https://colab.research.google.com/")
        logger.info("2. Upload the notebook file:")
        logger.info(f"    {self.colab_notebook_path}")
        logger.info("3. Enable GPU runtime: Runtime → Change runtime type → GPU")
        logger.info("4. Run all cells in the notebook to train PhoBERT")
        logger.info("5. Copy the ngrok URL when training completes")
        logger.info("6. Configure the URL using the API endpoint below")
        logger.info("="*80)
        logger.info(f" Configure Colab URL: POST http://localhost:{self.api_port}/api/config/colab")
        logger.info(f" Check training status: GET http://localhost:{self.api_port}/api/training/status")
        logger.info(f" Mark training complete: POST http://localhost:{self.api_port}/api/training/complete")
        logger.info("="*80)
    
    def wait_for_training_completion(self):
        """Wait for PhoBERT training to complete"""
        logger.info("⏳ Waiting for PhoBERT training completion...")
        logger.info(" The API will block pipeline requests until training is complete")
        
        while True:
            status = self.check_training_status()
            
            if status.get("can_use_pipelines", False):
                logger.info(" PhoBERT training completed! Pipelines are now ready.")
                self.training_complete = True
                break
            
            # Show current status
            training_status = status.get("training_status", {})
            if training_status.get("is_training", False):
                progress = training_status.get("training_progress", 0)
                logger.info(f" Training in progress: {progress:.1f}%")
            else:
                logger.info("⏸ Training not started yet. Please start training in Colab.")
            
            time.sleep(30)  # Check every 30 seconds
    
    def test_pipelines(self) -> bool:
        """Test both pipelines with sample queries"""
        logger.info("🧪 Testing both pipelines...")
        
        test_queries = [
            "Tìm điện thoại Samsung",
            "Cho tôi xem laptop Apple",
            "Sản phẩm giá rẻ dưới 10 triệu"
        ]
        
        success_count = 0
        
        for query in test_queries:
            try:
                logger.info(f"Testing: '{query}'")
                
                response = requests.post(
                    f"http://localhost:{self.api_port}/api/search",
                    json={"query": query, "pipeline": "both"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    p1_success = data.get("pipeline1_result", {}).get("success", False)
                    p2_success = data.get("pipeline2_result", {}).get("success", False)
                    
                    logger.info(f"  Pipeline 1: {'' if p1_success else ''}")
                    logger.info(f"  Pipeline 2: {'' if p2_success else ''}")
                    
                    if p1_success and p2_success:
                        success_count += 1
                else:
                    logger.error(f"  API Error: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"  Test failed: {e}")
        
        success_rate = success_count / len(test_queries)
        logger.info(f" Pipeline test results: {success_count}/{len(test_queries)} ({success_rate*100:.1f}%)")
        
        return success_rate >= 0.5  # At least 50% success rate
    
    def display_system_status(self):
        """Display comprehensive system status"""
        logger.info("\n" + "="*80)
        logger.info(" SYSTEM STATUS")
        logger.info("="*80)
        
        try:
            # API Health
            health_response = requests.get(f"http://localhost:{self.api_port}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                logger.info("🟢 API Server: Healthy")
                logger.info(f"   Database: {health_data.get('database', {}).get('connected', False)}")
                logger.info(f"   Products: {health_data.get('database', {}).get('products_count', 0)}")
            else:
                logger.info(" API Server: Unhealthy")
        except:
            logger.info(" API Server: Not responding")
        
        # Training Status
        training_status = self.check_training_status()
        status = training_status.get("training_status", {})
        
        logger.info(f"🤖 PhoBERT Training: {' Complete' if status.get('training_complete') else '⏳ Pending'}")
        logger.info(f" Models Ready: {' Yes' if status.get('model_ready') else ' No'}")
        logger.info(f" Pipelines Available: {' Yes' if training_status.get('can_use_pipelines') else ' No'}")
        
        logger.info("="*80)
        logger.info(" Available Endpoints:")
        logger.info(f"   API Documentation: http://localhost:{self.api_port}/docs")
        logger.info(f"   Health Check: http://localhost:{self.api_port}/health")
        logger.info(f"   Search API: http://localhost:{self.api_port}/api/search")
        logger.info(f"   Training Status: http://localhost:{self.api_port}/api/training/status")
        logger.info("="*80)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.api_process:
            logger.info("🧹 Stopping API server...")
            self.api_process.terminate()
            self.api_process.wait()
    
    def run(self):
        """Main execution flow"""
        try:
            logger.info(" Starting Vietnamese NL2SQL System")
            logger.info("="*80)
            
            # Step 1: Check dependencies
            if not self.check_dependencies():
                return False
            
            # Step 2: Setup database
            if not self.setup_database():
                return False
            
            # Step 3: Start API server
            if not self.start_api_server():
                return False
            
            # Step 4: Display training instructions
            self.display_colab_instructions()
            
            # Step 5: Wait for training completion
            self.wait_for_training_completion()
            
            # Step 6: Test pipelines
            if self.test_pipelines():
                logger.info(" System startup completed successfully!")
            else:
                logger.warning(" Some pipeline tests failed, but system is operational")
            
            # Step 7: Display final status
            self.display_system_status()
            
            # Keep running
            logger.info(" System running. Press Ctrl+C to stop.")
            while True:
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("⏹ Shutdown requested by user")
        except Exception as e:
            logger.error(f" System error: {e}")
        finally:
            self.cleanup()

def main():
    """Entry point"""
    coordinator = SystemCoordinator()
    coordinator.run()

if __name__ == "__main__":
    main()
