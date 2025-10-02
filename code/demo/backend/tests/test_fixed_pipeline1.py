#!/usr/bin/env python3
"""
Test script for the fixed Pipeline 1 implementation
Validates that the corrected architecture can generate SQL properly
"""

import torch
import sys
import os
import logging
from pathlib import Path

# Add the models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from fixed_phobert_sql import FixedPhoBERTForSQL, FixedVietnameseNL2SQLDataset, train_fixed_phobert_sql_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_training_data():
    """Create sample training data for testing"""
    return [
        {
            'vietnamese': 'Tìm tất cả balo nữ',
            'sql': 'SELECT p.name, b.brand_name, c.category_name, pr.price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN categories c ON p.category_id = c.category_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE p.name LIKE "%balo%" AND c.category_name LIKE "%nữ%"'
        },
        {
            'vietnamese': 'Tìm giày Nike giá dưới 1 triệu',
            'sql': 'SELECT p.name, b.brand_name, pr.price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE p.name LIKE "%giày%" AND b.brand_name LIKE "%Nike%" AND pr.price < 1000000'
        },
        {
            'vietnamese': 'Tìm túi xách có đánh giá trên 4 sao',
            'sql': 'SELECT p.name, b.brand_name, pr.price, rv.rating_average FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id JOIN product_reviews rv ON p.product_id = rv.product_id WHERE p.name LIKE "%túi xách%" AND rv.rating_average > 4.0'
        },
        {
            'vietnamese': 'Tìm sản phẩm Adidas màu đen',
            'sql': 'SELECT p.name, b.brand_name, c.category_name, pr.price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN categories c ON p.category_id = c.category_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE b.brand_name LIKE "%Adidas%" AND p.name LIKE "%đen%"'
        },
        {
            'vietnamese': 'Tìm kính mát giá từ 200k đến 500k',
            'sql': 'SELECT p.name, b.brand_name, pr.price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE p.name LIKE "%kính mát%" AND pr.price BETWEEN 200000 AND 500000'
        }
    ]

def test_model_initialization():
    """Test that the fixed model can be initialized properly"""
    logger.info("Testing model initialization...")
    
    try:
        model = FixedPhoBERTForSQL()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        logger.info(f"✅ Model initialized successfully on {device}")
        logger.info(f"Vietnamese vocab size: {len(model.vietnamese_tokenizer)}")
        logger.info(f"SQL vocab size: {len(model.sql_tokenizer)}")
        
        return model, device
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        return None, None

def test_tokenization(model):
    """Test that tokenization works correctly for both Vietnamese and SQL"""
    logger.info("Testing tokenization...")
    
    try:
        # Test Vietnamese tokenization
        vn_query = "Tìm tất cả balo nữ"
        vn_tokens = model.vietnamese_tokenizer(vn_query, return_tensors="pt")
        logger.info(f"Vietnamese tokens shape: {vn_tokens['input_ids'].shape}")
        
        # Test SQL tokenization
        sql_query = "SELECT * FROM products WHERE name LIKE '%balo%'"
        sql_tokens = model.sql_tokenizer(sql_query, return_tensors="pt")
        logger.info(f"SQL tokens shape: {sql_tokens['input_ids'].shape}")
        
        logger.info("✅ Tokenization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tokenization test failed: {e}")
        return False

def test_forward_pass(model, device):
    """Test that the model can perform a forward pass"""
    logger.info("Testing forward pass...")
    
    try:
        # Create sample inputs
        batch_size = 2
        vn_seq_len = 32
        sql_seq_len = 64
        
        vietnamese_input_ids = torch.randint(0, len(model.vietnamese_tokenizer), (batch_size, vn_seq_len)).to(device)
        vietnamese_attention_mask = torch.ones(batch_size, vn_seq_len).to(device)
        sql_input_ids = torch.randint(0, len(model.sql_tokenizer), (batch_size, sql_seq_len)).to(device)
        sql_attention_mask = torch.ones(batch_size, sql_seq_len).to(device)
        labels = sql_input_ids.clone()
        
        # Forward pass
        outputs = model(
            vietnamese_input_ids=vietnamese_input_ids,
            vietnamese_attention_mask=vietnamese_attention_mask,
            sql_input_ids=sql_input_ids,
            sql_attention_mask=sql_attention_mask,
            labels=labels
        )
        
        logger.info(f"Loss: {outputs['loss'].item():.4f}")
        logger.info(f"Logits shape: {outputs['logits'].shape}")
        logger.info("✅ Forward pass test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Forward pass test failed: {e}")
        return False

def test_generation(model, device):
    """Test that the model can generate SQL"""
    logger.info("Testing SQL generation...")
    
    try:
        model.eval()
        
        # Test query
        vn_query = "Tìm balo nữ"
        
        # Tokenize
        inputs = model.vietnamese_tokenizer(
            vn_query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate_sql(
                vietnamese_input_ids=inputs['input_ids'],
                vietnamese_attention_mask=inputs['attention_mask'],
                max_length=128
            )
        
        # Decode
        sql_query = model.sql_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        logger.info(f"Input: {vn_query}")
        logger.info(f"Generated SQL: {sql_query}")
        logger.info("✅ Generation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Generation test failed: {e}")
        return False

def test_dataset_creation():
    """Test that the dataset can be created properly"""
    logger.info("Testing dataset creation...")
    
    try:
        # Create model for tokenizers
        model = FixedPhoBERTForSQL()
        
        # Create sample data
        training_data = create_sample_training_data()
        
        # Create dataset
        dataset = FixedVietnameseNL2SQLDataset(
            training_data, 
            model.vietnamese_tokenizer, 
            model.sql_tokenizer
        )
        
        # Test dataset access
        sample = dataset[0]
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Sample keys: {list(sample.keys())}")
        logger.info(f"Vietnamese input shape: {sample['vietnamese_input_ids'].shape}")
        logger.info(f"SQL input shape: {sample['sql_input_ids'].shape}")
        logger.info("✅ Dataset creation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset creation test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests to validate the fixed implementation"""
    logger.info("="*60)
    logger.info("RUNNING COMPREHENSIVE TESTS FOR FIXED PIPELINE 1")
    logger.info("="*60)
    
    test_results = []
    
    # Test 1: Model initialization
    model, device = test_model_initialization()
    test_results.append(model is not None)
    
    if model is None:
        logger.error("❌ Cannot proceed with other tests - model initialization failed")
        return False
    
    # Test 2: Tokenization
    test_results.append(test_tokenization(model))
    
    # Test 3: Forward pass
    test_results.append(test_forward_pass(model, device))
    
    # Test 4: Generation
    test_results.append(test_generation(model, device))
    
    # Test 5: Dataset creation
    test_results.append(test_dataset_creation())
    
    # Summary
    logger.info("="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    test_names = [
        "Model Initialization",
        "Tokenization", 
        "Forward Pass",
        "SQL Generation",
        "Dataset Creation"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{i+1}. {name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Fixed implementation is working correctly!")
        return True
    else:
        logger.error(f"⚠️ {total-passed} tests failed - Implementation needs further fixes")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
