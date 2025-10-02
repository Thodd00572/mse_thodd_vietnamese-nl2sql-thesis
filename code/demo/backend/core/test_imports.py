#!/usr/bin/env python3
"""
Test script to verify import paths are working correctly
"""

import sys
import os

# Add both core directory and backend root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_root = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(backend_root)

print(f"Current directory: {current_dir}")
print(f"Backend root: {backend_root}")
print(f"Python path: {sys.path}")

try:
    print("\nTesting imports...")
    
    # Test API imports
    from api import routes
    from api import sample_query_routes
    print("✅ API imports successful")
    
    # Test database import
    from database.db_manager_normalized import DatabaseManager
    print("✅ Database import successful")
    
    # Test models import
    from models.pipelines import pipeline1, pipeline2
    print("✅ Models import successful")
    
    print("\n🎉 All imports successful! The server should start correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nAvailable modules in backend root:")
    for item in os.listdir(backend_root):
        if os.path.isdir(os.path.join(backend_root, item)) and not item.startswith('.'):
            print(f"  - {item}")

except Exception as e:
    print(f"❌ Other error: {e}")
