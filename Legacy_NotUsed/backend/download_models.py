#!/usr/bin/env python3
"""
Download required models for MSE Thesis Vietnamese NL2SQL project
"""

import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_sqlcoder_model():
    """Download SQLCoder-7B-2 model from Hugging Face"""
    try:
        logger.info("Downloading SQLCoder-7B-2 model...")
        
        # Create cache directory
        cache_dir = os.path.join(os.path.dirname(__file__), "models", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download the model using transformers
        model_name = "defog/sqlcoder-7b-2"
        
        logger.info(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        logger.info(f"Downloading model for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("SQLCoder-7B-2 model downloaded successfully!")
        logger.info(f"Model cached in: {cache_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading SQLCoder model: {e}")
        return False

def download_vietnamese_english_model():
    """Download Vietnamese-English translation model"""
    try:
        logger.info("Downloading Vietnamese-English translation model...")
        
        cache_dir = os.path.join(os.path.dirname(__file__), "models", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_name = "Helsinki-NLP/opus-mt-vi-en"
        
        logger.info(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info(f"Downloading model for {model_name}...")
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info("Vietnamese-English translation model downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading Vietnamese-English model: {e}")
        return False

def download_phobert_model():
    """Download PhoBERT model"""
    try:
        logger.info("Downloading PhoBERT model...")
        
        cache_dir = os.path.join(os.path.dirname(__file__), "models", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_name = "vinai/phobert-base"
        
        logger.info(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info(f"Downloading model for {model_name}...")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info("PhoBERT model downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading PhoBERT model: {e}")
        return False

def main():
    """Download all required models"""
    logger.info("Starting model download process...")
    
    results = {
        "sqlcoder": download_sqlcoder_model(),
        "vietnamese_english": download_vietnamese_english_model(),
        "phobert": download_phobert_model()
    }
    
    success_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"\n=== DOWNLOAD SUMMARY ===")
    logger.info(f"Successfully downloaded: {success_count}/{total_count} models")
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"  {model_name}: {status}")
    
    if success_count == total_count:
        logger.info("\n🎉 All models downloaded successfully!")
        logger.info("You can now run the application with full model support.")
    else:
        logger.warning(f"\n⚠️  {total_count - success_count} models failed to download.")
        logger.warning("Some pipelines may fall back to rule-based methods.")
    
    return success_count == total_count

if __name__ == "__main__":
    main()
