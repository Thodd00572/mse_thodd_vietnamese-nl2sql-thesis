#!/usr/bin/env python3
"""
Quick fix for Pipeline 2 response parsing issue
"""

import json
import requests

def test_pipeline2_direct():
    """Test Pipeline 2 directly via Colab API"""
    print("🧪 Testing Pipeline 2 directly via Colab API...")
    
    url = "https://abnormally-direct-rhino.ngrok-free.app/pipeline2"
    payload = {"query": "tìm áo thun nam"}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print("✅ Colab API Response:")
        print(f"  SQL Query: {data.get('sql_query', 'N/A')}")
        print(f"  English Translation: {data.get('english_translation', 'N/A')}")
        print(f"  Success: {data.get('success', 'N/A')}")
        print(f"  Execution Time: {data.get('execution_time', 'N/A')}s")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_local_backend():
    """Test local backend Pipeline 2"""
    print("\n🧪 Testing local backend Pipeline 2...")
    
    url = "http://localhost:8000/api/search"
    payload = {"query": "tìm áo thun nam", "pipeline": "pipeline2"}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print("📊 Local Backend Response:")
        print(f"  Pipeline 2 Success: {data.get('pipeline2_result', {}).get('success', 'N/A')}")
        print(f"  SQL Query: {data.get('pipeline2_result', {}).get('sql_query', 'N/A')}")
        print(f"  English Query: {data.get('pipeline2_result', {}).get('english_query', 'N/A')}")
        print(f"  Error: {data.get('pipeline2_result', {}).get('error', 'N/A')}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🔍 Pipeline 2 Response Debugging")
    print("=" * 50)
    
    # Test Colab API directly
    colab_data = test_pipeline2_direct()
    
    # Test local backend
    local_data = test_local_backend()
    
    # Compare results
    if colab_data and local_data:
        print("\n📋 Comparison:")
        print(f"Colab SQL: {colab_data.get('sql_query', 'N/A')}")
        print(f"Local SQL: {local_data.get('pipeline2_result', {}).get('sql_query', 'N/A')}")
        
        if colab_data.get('sql_query') and not local_data.get('pipeline2_result', {}).get('sql_query'):
            print("❌ Issue: Colab returns SQL but local backend doesn't extract it")
        elif colab_data.get('sql_query') == local_data.get('pipeline2_result', {}).get('sql_query'):
            print("✅ Success: SQL queries match!")
