#!/usr/bin/env python3
"""
Download SQLCoder-7B-2 model with retry logic and better error handling
"""

import os
import logging
import time
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_sqlcoder_with_retry(max_retries=3):
    """Download SQLCoder model with retry logic"""
    cache_dir = os.path.join(os.path.dirname(__file__), "models", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    model_name = "defog/sqlcoder-7b-2"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Downloading SQLCoder-7B-2...")
            
            # Method 1: Try downloading with snapshot_download (more reliable for large models)
            logger.info("Using snapshot_download method...")
            local_dir = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False,
                timeout=300  # 5 minutes timeout
            )
            
            logger.info(f"SQLCoder model downloaded successfully to: {local_dir}")
            return True, local_dir
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # Progressive backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error("All download attempts failed")
    
    return False, None

def download_individual_files():
    """Download SQLCoder files individually as fallback"""
    try:
        logger.info("Trying individual file download as fallback...")
        cache_dir = os.path.join(os.path.dirname(__file__), "models", "cache")
        model_name = "defog/sqlcoder-7b-2"
        
        # Essential files to download
        files_to_download = [
            "config.json",
            "tokenizer_config.json", 
            "tokenizer.json",
            "tokenizer.model",
            "special_tokens_map.json"
        ]
        
        for filename in files_to_download:
            try:
                logger.info(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=model_name,
                    filename=filename,
                    cache_dir=cache_dir,
                    resume_download=True
                )
                logger.info(f"✅ Downloaded {filename}")
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")
        
        # Try to download at least one model shard
        try:
            logger.info("Downloading model shard 1...")
            hf_hub_download(
                repo_id=model_name,
                filename="model-00001-of-00003.safetensors",
                cache_dir=cache_dir,
                resume_download=True
            )
            logger.info("✅ Downloaded model shard 1")
        except Exception as e:
            logger.warning(f"Failed to download model shard: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Individual file download failed: {e}")
        return False

def main():
    """Main download function"""
    logger.info("Starting SQLCoder-7B-2 download...")
    
    # Try full download first
    success, local_dir = download_sqlcoder_with_retry()
    
    if not success:
        logger.info("Full download failed, trying individual files...")
        success = download_individual_files()
    
    if success:
        logger.info("🎉 SQLCoder model download completed!")
        
        # Verify the download
        cache_dir = os.path.join(os.path.dirname(__file__), "models", "cache")
        logger.info(f"Model files cached in: {cache_dir}")
        
        # List downloaded files
        try:
            for root, dirs, files in os.walk(cache_dir):
                if "sqlcoder" in root.lower():
                    logger.info(f"Found SQLCoder files in: {root}")
                    for file in files[:5]:  # Show first 5 files
                        logger.info(f"  - {file}")
                    if len(files) > 5:
                        logger.info(f"  ... and {len(files) - 5} more files")
        except Exception as e:
            logger.warning(f"Could not list files: {e}")
            
    else:
        logger.error("❌ SQLCoder model download failed completely")
        logger.info("Pipeline 2 will fall back to rule-based translation")
    
    return success

if __name__ == "__main__":
    main()
