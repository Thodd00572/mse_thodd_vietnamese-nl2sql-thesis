#!/usr/bin/env python3
"""
PhoBERT Fine-tuning for Vietnamese NL2SQL
Google Colab Training Implementation

Author: MSE14 Duong Dinh Tho
© Copyright by MSE 14 Duong Dinh Tho, 2025
"""

# Cell 1 - Environment Setup
import subprocess
import sys

print("Installing required packages for PhoBERT fine-tuning...")

packages = [
    "transformers",
    "torch",
    "datasets", 
    "accelerate",
    "evaluate",
    "rouge-score",
    "sacrebleu",
    "wandb",
    "tensorboard"
]

for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print(f"✅ {package}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")

print("Package installation completed")

# Cell 2 - Imports and Setup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
import pandas as pd
import numpy as np
import json
import random
from typing import Dict, List, Any, Tuple
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Device: {device}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 3 - Training Data Generation
class VietnameseNL2SQLDataGenerator:
    """Generate Vietnamese NL2SQL training pairs"""
    
    def __init__(self):
        self.schema = {
            "products": {
                "columns": ["id", "name", "brand", "category", "price", "rating", "review_count"],
                "types": ["INTEGER", "TEXT", "TEXT", "TEXT", "REAL", "REAL", "INTEGER"]
            }
        }
        
        self.vietnamese_templates = {
            "product_search": [
                "Tìm {product_type}",
                "Cho tôi xem {product_type}",
                "Hiển thị {product_type}",
                "Tôi muốn mua {product_type}"
            ],
            "brand_filter": [
                "Tìm {product_type} {brand}",
                "Cho tôi xem {product_type} của {brand}",
                "Hiển thị sản phẩm {brand}",
                "{product_type} {brand} có gì"
            ],
            "price_filter": [
                "Tìm {product_type} dưới {price} triệu",
                "{product_type} giá rẻ dưới {price} triệu",
                "Sản phẩm dưới {price} triệu",
                "{product_type} không quá {price} triệu"
            ],
            "price_range": [
                "Tìm {product_type} từ {min_price} đến {max_price} triệu",
                "{product_type} trong khoảng {min_price}-{max_price} triệu",
                "Sản phẩm giá từ {min_price} triệu đến {max_price} triệu"
            ],
            "rating_filter": [
                "Tìm {product_type} rating trên {rating} sao",
                "{product_type} đánh giá cao trên {rating} sao",
                "Sản phẩm tốt trên {rating} sao"
            ],
            "top_products": [
                "Top {limit} {product_type} tốt nhất",
                "{limit} {product_type} bán chạy nhất",
                "Top {limit} sản phẩm rating cao nhất"
            ],
            "count_products": [
                "Có bao nhiêu {product_type}",
                "Đếm số lượng {product_type}",
                "Tổng cộng có bao nhiêu sản phẩm"
            ]
        }
        
        self.sql_templates = {
            "product_search": "SELECT * FROM products WHERE category = '{category}'",
            "brand_filter": "SELECT * FROM products WHERE brand = '{brand}' AND category = '{category}'",
            "price_filter": "SELECT * FROM products WHERE price <= {price} AND category = '{category}'",
            "price_range": "SELECT * FROM products WHERE price BETWEEN {min_price} AND {max_price} AND category = '{category}'",
            "rating_filter": "SELECT * FROM products WHERE rating >= {rating} AND category = '{category}'",
            "top_products": "SELECT * FROM products WHERE category = '{category}' ORDER BY rating DESC LIMIT {limit}",
            "count_products": "SELECT COUNT(*) FROM products WHERE category = '{category}'"
        }
        
        self.entities = {
            "product_types": ["điện thoại", "laptop", "túi xách", "tai nghe", "đồng hồ"],
            "brands": ["Samsung", "Apple", "Sony", "Dell", "HP", "Xiaomi"],
            "categories": ["điện thoại", "laptop", "túi xách", "tai nghe", "đồng hồ"],
            "prices": [5, 10, 15, 20, 25, 30],
            "ratings": [3.5, 4.0, 4.5],
            "limits": [5, 10, 15, 20]
        }
    
    def generate_training_pairs(self, num_samples: int = 1000) -> List[Dict[str, str]]:
        """Generate Vietnamese NL2SQL training pairs"""
        training_data = []
        
        for _ in range(num_samples):
            # Randomly select template type
            template_type = random.choice(list(self.vietnamese_templates.keys()))
            
            # Generate Vietnamese query
            vn_template = random.choice(self.vietnamese_templates[template_type])
            sql_template = self.sql_templates[template_type]
            
            # Fill templates with random entities
            entities = self._generate_random_entities(template_type)
            
            try:
                vietnamese_query = vn_template.format(**entities)
                sql_query = sql_template.format(**entities)
                
                training_data.append({
                    "vietnamese_query": vietnamese_query,
                    "sql_query": sql_query,
                    "template_type": template_type
                })
            except KeyError:
                continue  # Skip if template formatting fails
        
        return training_data
    
    def _generate_random_entities(self, template_type: str) -> Dict[str, Any]:
        """Generate random entities for template filling"""
        entities = {}
        
        # Common entities
        entities["product_type"] = random.choice(self.entities["product_types"])
        entities["category"] = entities["product_type"]  # Map product type to category
        
        if template_type == "brand_filter":
            entities["brand"] = random.choice(self.entities["brands"])
        
        elif template_type == "price_filter":
            entities["price"] = random.choice(self.entities["prices"]) * 1000000  # Convert to VND
        
        elif template_type == "price_range":
            prices = random.sample(self.entities["prices"], 2)
            entities["min_price"] = min(prices) * 1000000
            entities["max_price"] = max(prices) * 1000000
        
        elif template_type == "rating_filter":
            entities["rating"] = random.choice(self.entities["ratings"])
        
        elif template_type == "top_products":
            entities["limit"] = random.choice(self.entities["limits"])
        
        return entities

# Generate training data
print("🔄 Generating Vietnamese NL2SQL training data...")
data_generator = VietnameseNL2SQLDataGenerator()
training_pairs = data_generator.generate_training_pairs(2000)
print(f"✅ Generated {len(training_pairs)} training pairs")

# Sample data
for i in range(3):
    print(f"\nSample {i+1}:")
    print(f"Vietnamese: {training_pairs[i]['vietnamese_query']}")
    print(f"SQL: {training_pairs[i]['sql_query']}")

# Cell 4 - Custom PhoBERT Model for SQL Generation
class PhoBERTForSQL(nn.Module):
    """PhoBERT with SQL generation head"""
    
    def __init__(self, model_name="vinai/phobert-base", max_sql_length=128):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # SQL generation head
        self.sql_head = nn.Sequential(
            nn.Linear(self.phobert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.tokenizer.vocab_size)
        )
        
        self.max_sql_length = max_sql_length
        
    def forward(self, input_ids, attention_mask=None, sql_labels=None):
        # PhoBERT encoding
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Use CLS token for SQL generation
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Generate SQL logits
        sql_logits = self.sql_head(cls_output)  # [batch_size, vocab_size]
        
        loss = None
        if sql_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Repeat logits for sequence length
            sql_logits_expanded = sql_logits.unsqueeze(1).repeat(1, sql_labels.size(1), 1)
            loss = loss_fct(sql_logits_expanded.view(-1, sql_logits_expanded.size(-1)), 
                          sql_labels.view(-1))
        
        return {
            "loss": loss,
            "logits": sql_logits,
            "hidden_states": sequence_output
        }
    
    def generate_sql(self, vietnamese_query: str, max_length: int = 128):
        """Generate SQL from Vietnamese query"""
        self.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            vietnamese_query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        with torch.no_grad():
            outputs = self.forward(**inputs)
            logits = outputs["logits"]
            
            # Simple greedy decoding (can be improved with beam search)
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode to SQL
            sql_tokens = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
        return sql_tokens

# Cell 5 - Dataset Class
class VietnameseNL2SQLDataset(Dataset):
    """Dataset for Vietnamese NL2SQL training"""
    
    def __init__(self, data_pairs: List[Dict], tokenizer, max_length: int = 256):
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        pair = self.data_pairs[idx]
        
        # Tokenize Vietnamese query
        vietnamese_encoding = self.tokenizer(
            pair["vietnamese_query"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize SQL query
        sql_encoding = self.tokenizer(
            pair["sql_query"],
            truncation=True,
            padding="max_length", 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": vietnamese_encoding["input_ids"].squeeze(),
            "attention_mask": vietnamese_encoding["attention_mask"].squeeze(),
            "sql_labels": sql_encoding["input_ids"].squeeze()
        }

# Cell 6 - Training Setup
def setup_training():
    """Setup model and training components"""
    
    # Initialize model
    model = PhoBERTForSQL()
    model.to(device)
    
    # Create datasets
    train_size = int(0.8 * len(training_pairs))
    train_data = training_pairs[:train_size]
    val_data = training_pairs[train_size:]
    
    train_dataset = VietnameseNL2SQLDataset(train_data, model.tokenizer)
    val_dataset = VietnameseNL2SQLDataset(val_data, model.tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./phobert-sql-results",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb for now
        dataloader_pin_memory=False
    )
    
    # Custom trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    return model, trainer, train_dataset, val_dataset

# Cell 7 - Training Execution
def train_phobert_sql():
    """Execute PhoBERT fine-tuning"""
    
    print("🚀 Starting PhoBERT fine-tuning for Vietnamese NL2SQL...")
    
    # Setup training
    model, trainer, train_dataset, val_dataset = setup_training()
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"🔥 Device: {device}")
    
    # Start training
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time/60:.2f} minutes")
        
        # Save model
        model.save_pretrained("./phobert-sql-final")
        model.tokenizer.save_pretrained("./phobert-sql-final")
        
        print("💾 Model saved to ./phobert-sql-final")
        
        return model, trainer
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, None

# Cell 8 - Testing and Evaluation
def test_trained_model(model):
    """Test the trained PhoBERT-SQL model"""
    
    test_queries = [
        "Tìm điện thoại Samsung",
        "Laptop dưới 20 triệu",
        "Top 5 sản phẩm tốt nhất",
        "Có bao nhiêu túi xách"
    ]
    
    print("🧪 Testing trained model...")
    
    for query in test_queries:
        try:
            sql_result = model.generate_sql(query)
            print(f"\n📝 Query: {query}")
            print(f"🔍 Generated SQL: {sql_result}")
        except Exception as e:
            print(f"❌ Error generating SQL for '{query}': {e}")

# Cell 9 - Main Execution
if __name__ == "__main__":
    print("🎯 PhoBERT Vietnamese NL2SQL Training Pipeline")
    print("=" * 50)
    
    # Execute training
    trained_model, trainer = train_phobert_sql()
    
    if trained_model:
        # Test the model
        test_trained_model(trained_model)
        
        print("\n🎉 Training pipeline completed successfully!")
        print("📁 Model saved in ./phobert-sql-final")
        print("🔄 Ready for integration with main pipeline")
    else:
        print("\n❌ Training failed. Check logs for details.")

# Instructions for Google Colab
print("""
📋 GOOGLE COLAB INSTRUCTIONS:

1. 🔄 Run all cells in sequence
2. 🔥 Ensure GPU is enabled (Runtime > Change runtime type > GPU)
3. ⏱️  Training takes ~30-60 minutes on T4 GPU
4. 💾 Model will be saved to ./phobert-sql-final
5. 🔗 Download the model or upload to Google Drive for persistence

📊 MONITORING:
- Watch GPU memory usage
- Monitor training loss
- Check validation metrics

🎯 NEXT STEPS:
- Integrate trained model into main pipeline
- Replace rule-based SQL generation
- Evaluate against Pipeline 2
""")
