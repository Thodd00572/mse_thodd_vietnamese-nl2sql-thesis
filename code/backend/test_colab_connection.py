#!/usr/bin/env python3
"""
Test script to diagnose Colab server connection issues
"""

import requests
import json
import time
from colab_client import ColabAPIClient

def test_colab_connection():
    """Test connection to Colab server with detailed diagnostics"""
    
    print("🔍 Testing Colab Server Connection")
    print("=" * 50)
    
    # Initialize client
    client = ColabAPIClient()
    
    # Test different possible URLs
    test_urls = [
        "https://abnormally-direct-rhino.ngrok-free.app",
        "https://abnormally-direct-rhino.ngrok.io",
        # Add more potential URLs if needed
    ]
    
    for base_url in test_urls:
        print(f"\n🌐 Testing base URL: {base_url}")
        
        # Set the base URL
        client.set_base_url(base_url)
        
        # Test connection
        status = client.test_connection()
        
        print(f"📊 Connection Status:")
        print(f"  Base URL: {status['base_url']}")
        print(f"  Pipeline 1 URL: {status['pipeline1_url']}")
        print(f"  Pipeline 2 URL: {status['pipeline2_url']}")
        
        if status['health_check']:
            print(f"✅ Health Check: {status['health_check']['status']}")
            print(f"  Pipeline 1 Loaded: {status['health_check']['pipeline1_loaded']}")
            print(f"  Pipeline 2 Loaded: {status['health_check']['pipeline2_loaded']}")
            print(f"  GPU Available: {status['health_check']['server_info']['gpu_available']}")
            
            # If health check passes, test actual pipeline endpoints
            print(f"\n🧪 Testing Pipeline Endpoints...")
            
            # Test Pipeline 1
            try:
                test_query = "tìm áo thun"
                response = requests.post(
                    status['pipeline1_url'],
                    json={"query": test_query},
                    timeout=30
                )
                if response.status_code == 200:
                    print(f"✅ Pipeline 1 test successful")
                    result = response.json()
                    print(f"  SQL Query: {result.get('sql_query', 'N/A')}")
                else:
                    print(f"❌ Pipeline 1 test failed: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ Pipeline 1 test error: {e}")
            
            # Test Pipeline 2
            try:
                response = requests.post(
                    status['pipeline2_url'],
                    json={"query": test_query},
                    timeout=30
                )
                if response.status_code == 200:
                    print(f"✅ Pipeline 2 test successful")
                    result = response.json()
                    print(f"  English Query: {result.get('english_query', 'N/A')}")
                    print(f"  SQL Query: {result.get('sql_query', 'N/A')}")
                else:
                    print(f"❌ Pipeline 2 test failed: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ Pipeline 2 test error: {e}")
            
            return True  # Found working URL
            
        else:
            print(f"❌ Health Check Failed")
            for error in status['connection_errors']:
                print(f"  Error: {error}")
    
    print(f"\n❌ No working Colab server found")
    print(f"\n🔧 Troubleshooting Steps:")
    print(f"1. Check if Colab notebook is running")
    print(f"2. Verify ngrok tunnel is active")
    print(f"3. Check ngrok domain configuration")
    print(f"4. Ensure models loaded successfully in Colab")
    
    return False

def check_local_backend_config():
    """Check local backend configuration"""
    print(f"\n🔍 Checking Local Backend Configuration")
    print("=" * 50)
    
    client = ColabAPIClient()
    
    print(f"Current Configuration:")
    print(f"  Base URL: {client.base_url}")
    print(f"  Pipeline 1 URL: {client.pipeline1_url}")
    print(f"  Pipeline 2 URL: {client.pipeline2_url}")
    
    if not client.base_url:
        print(f"\n⚠️ No Colab URL configured in local backend!")
        print(f"💡 To fix this, you need to:")
        print(f"1. Get the ngrok URL from your Colab notebook")
        print(f"2. Configure it in your local backend using:")
        print(f"   client.set_base_url('https://your-ngrok-url.ngrok-free.app')")

if __name__ == "__main__":
    print("🚀 Colab Connection Diagnostic Tool")
    print("MSE Thesis 2025 - Vietnamese NL2SQL System")
    print()
    
    # Check local configuration first
    check_local_backend_config()
    
    # Test connection
    success = test_colab_connection()
    
    if success:
        print(f"\n✅ Connection test completed successfully!")
    else:
        print(f"\n❌ Connection test failed - see troubleshooting steps above")
