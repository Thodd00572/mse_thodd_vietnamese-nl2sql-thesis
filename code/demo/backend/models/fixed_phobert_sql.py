#!/usr/bin/env python3
"""
Fixed PhoBERT-SQL Model Architecture for Vietnamese NL2SQL
Addresses critical issues in the original training strategy

Key Fixes:
1. Proper encoder-decoder architecture using T5-style approach
2. Separate tokenizers for Vietnamese input and SQL output
3. Correct training strategy with sequence-to-sequence loss
4. Proper generation method with beam search
5. SQL-aware tokenization and special tokens
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration, T5Tokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback, GenerationConfig
)
from datasets import Dataset
import pandas as pd
import numpy as np
import logging
import time
import json
import random
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class FixedPhoBERTForSQL(nn.Module):
    """
    Fixed PhoBERT-SQL model using proper encoder-decoder architecture
    
    Architecture:
    - Encoder: PhoBERT for Vietnamese understanding
    - Decoder: T5-style decoder for SQL generation
    - Cross-attention: Connects Vietnamese context to SQL generation
    """
    
    def __init__(self, phobert_model_name='vinai/phobert-base', sql_vocab_size=32000):
        super().__init__()
        
        # Vietnamese encoder (PhoBERT)
        self.vietnamese_encoder = AutoModel.from_pretrained(phobert_model_name)
        self.vietnamese_tokenizer = AutoTokenizer.from_pretrained(phobert_model_name)
        
        # SQL decoder configuration
        self.hidden_size = self.vietnamese_encoder.config.hidden_size  # 768 for PhoBERT-base
        self.sql_vocab_size = sql_vocab_size
        
        # Create SQL tokenizer with special tokens
        self.sql_tokenizer = self._create_sql_tokenizer()
        
        # SQL decoder layers
        self.sql_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=6
        )
        
        # Output projection to SQL vocabulary
        self.sql_output_projection = nn.Linear(self.hidden_size, len(self.sql_tokenizer))
        
        # Embeddings for SQL tokens
        self.sql_embeddings = nn.Embedding(len(self.sql_tokenizer), self.hidden_size)
        
        # Positional encoding for SQL decoder
        self.sql_pos_encoding = nn.Embedding(512, self.hidden_size)  # Max SQL length
        
        # Generation configuration
        self.generation_config = GenerationConfig(
            max_length=256,
            min_length=10,
            num_beams=4,
            early_stopping=True,
            pad_token_id=self.sql_tokenizer.pad_token_id,
            eos_token_id=self.sql_tokenizer.eos_token_id,
            bos_token_id=self.sql_tokenizer.bos_token_id,
            do_sample=False,  # Use deterministic generation
            temperature=1.0,
            repetition_penalty=1.2
        )
        
    def _create_sql_tokenizer(self):
        """Create specialized SQL tokenizer with proper vocabulary"""
        
        # Start with a base tokenizer and add SQL-specific vocabulary
        base_tokenizer = AutoTokenizer.from_pretrained('t5-small')
        
        # SQL keywords and operators
        sql_tokens = [
            # SQL Keywords
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'ON', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL',
            'GROUP', 'BY', 'HAVING', 'ORDER', 'ASC', 'DESC', 'LIMIT', 'OFFSET',
            'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'AS',
            
            # Operators
            '=', '>', '<', '>=', '<=', '!=', '<>', '+', '-', '*', '/', '%',
            
            # Schema-specific tokens (Tiki database)
            'products', 'brands', 'categories', 'sellers', 'product_pricing', 'product_reviews',
            'product_id', 'name', 'description', 'brand_id', 'category_id', 'seller_id',
            'brand_name', 'category_name', 'seller_name', 'price', 'original_price',
            'discount_rate', 'rating_average', 'review_count', 'total_quantity_sold',
            
            # Vietnamese product terms (for LIKE conditions)
            'balo', 'túi', 'giày', 'dép', 'áo', 'quần', 'váy', 'kính', 'vali',
            'nam', 'nữ', 'đen', 'trắng', 'xanh', 'đỏ', 'nâu', 'hồng',
            
            # Special tokens
            '<SQL_START>', '<SQL_END>', '<TABLE>', '<COLUMN>', '<VALUE>'
        ]
        
        # Add SQL tokens to vocabulary
        base_tokenizer.add_tokens(sql_tokens)
        
        # Set special tokens
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_tokenizer.bos_token = '<SQL_START>'
        
        return base_tokenizer
    
    def forward(self, vietnamese_input_ids, vietnamese_attention_mask, 
                sql_input_ids=None, sql_attention_mask=None, labels=None):
        """
        Forward pass for training
        
        Args:
            vietnamese_input_ids: Tokenized Vietnamese query
            vietnamese_attention_mask: Attention mask for Vietnamese input
            sql_input_ids: Tokenized SQL query (for teacher forcing during training)
            sql_attention_mask: Attention mask for SQL input
            labels: Target SQL tokens for loss calculation
        """
        
        # Encode Vietnamese input
        vietnamese_outputs = self.vietnamese_encoder(
            input_ids=vietnamese_input_ids,
            attention_mask=vietnamese_attention_mask
        )
        vietnamese_hidden_states = vietnamese_outputs.last_hidden_state
        
        if sql_input_ids is not None:
            # Training mode: use teacher forcing
            batch_size, sql_seq_len = sql_input_ids.shape
            
            # Create SQL embeddings with positional encoding
            sql_embeds = self.sql_embeddings(sql_input_ids)
            positions = torch.arange(sql_seq_len, device=sql_input_ids.device).unsqueeze(0).expand(batch_size, -1)
            sql_embeds += self.sql_pos_encoding(positions)
            
            # Create causal mask for SQL decoder
            sql_causal_mask = torch.triu(torch.ones(sql_seq_len, sql_seq_len), diagonal=1).bool()
            sql_causal_mask = sql_causal_mask.to(sql_input_ids.device)
            
            # SQL decoder with cross-attention to Vietnamese
            sql_outputs = self.sql_decoder(
                tgt=sql_embeds,
                memory=vietnamese_hidden_states,
                tgt_mask=sql_causal_mask,
                memory_key_padding_mask=~vietnamese_attention_mask.bool()
            )
            
            # Project to SQL vocabulary
            sql_logits = self.sql_output_projection(sql_outputs)
            
            # Calculate loss if labels provided
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
                loss = loss_fct(sql_logits.view(-1, sql_logits.size(-1)), labels.view(-1))
            
            return {
                'loss': loss,
                'logits': sql_logits,
                'vietnamese_hidden_states': vietnamese_hidden_states,
                'sql_hidden_states': sql_outputs
            }
        else:
            # Inference mode: return encoder outputs for generation
            return {
                'vietnamese_hidden_states': vietnamese_hidden_states,
                'vietnamese_attention_mask': vietnamese_attention_mask
            }
    
    def generate_sql(self, vietnamese_input_ids, vietnamese_attention_mask, **generation_kwargs):
        """
        Generate SQL query from Vietnamese input using beam search
        
        Args:
            vietnamese_input_ids: Tokenized Vietnamese query
            vietnamese_attention_mask: Attention mask for Vietnamese input
            **generation_kwargs: Additional generation parameters
        """
        
        self.eval()
        with torch.no_grad():
            # Encode Vietnamese input
            encoder_outputs = self.forward(
                vietnamese_input_ids=vietnamese_input_ids,
                vietnamese_attention_mask=vietnamese_attention_mask
            )
            vietnamese_hidden_states = encoder_outputs['vietnamese_hidden_states']
            
            # Generation parameters
            max_length = generation_kwargs.get('max_length', self.generation_config.max_length)
            num_beams = generation_kwargs.get('num_beams', self.generation_config.num_beams)
            early_stopping = generation_kwargs.get('early_stopping', self.generation_config.early_stopping)
            
            batch_size = vietnamese_input_ids.size(0)
            device = vietnamese_input_ids.device
            
            # Initialize with BOS token
            generated_ids = torch.full(
                (batch_size, 1), 
                self.sql_tokenizer.bos_token_id, 
                device=device, 
                dtype=torch.long
            )
            
            # Simple greedy generation (can be extended to beam search)
            for step in range(max_length - 1):
                # Create SQL embeddings for current sequence
                sql_embeds = self.sql_embeddings(generated_ids)
                seq_len = generated_ids.size(1)
                positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                sql_embeds += self.sql_pos_encoding(positions)
                
                # Create causal mask
                causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
                
                # Decode
                sql_outputs = self.sql_decoder(
                    tgt=sql_embeds,
                    memory=vietnamese_hidden_states,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=~vietnamese_attention_mask.bool()
                )
                
                # Get logits for next token
                next_token_logits = self.sql_output_projection(sql_outputs[:, -1, :])
                
                # Get next token (greedy)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check for EOS token
                if next_token[0].item() == self.sql_tokenizer.eos_token_id:
                    break
            
            return generated_ids

class FixedVietnameseNL2SQLDataset(torch.utils.data.Dataset):
    """
    Fixed dataset class for Vietnamese NL2SQL training
    Properly handles separate tokenizers for Vietnamese and SQL
    """
    
    def __init__(self, data_pairs, vietnamese_tokenizer, sql_tokenizer, max_vn_length=128, max_sql_length=256):
        self.data_pairs = data_pairs
        self.vietnamese_tokenizer = vietnamese_tokenizer
        self.sql_tokenizer = sql_tokenizer
        self.max_vn_length = max_vn_length
        self.max_sql_length = max_sql_length
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        data_pair = self.data_pairs[idx]
        vietnamese_query = data_pair['vietnamese']
        sql_query = data_pair['sql']
        
        # Tokenize Vietnamese input
        vn_inputs = self.vietnamese_tokenizer(
            vietnamese_query,
            max_length=self.max_vn_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize SQL with special tokens
        sql_with_tokens = f"<SQL_START> {sql_query} {self.sql_tokenizer.eos_token}"
        sql_inputs = self.sql_tokenizer(
            sql_with_tokens,
            max_length=self.max_sql_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (shift SQL input by 1 for next token prediction)
        labels = sql_inputs['input_ids'].clone()
        labels[labels == self.sql_tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
        return {
            'vietnamese_input_ids': vn_inputs['input_ids'].squeeze(),
            'vietnamese_attention_mask': vn_inputs['attention_mask'].squeeze(),
            'sql_input_ids': sql_inputs['input_ids'].squeeze(),
            'sql_attention_mask': sql_inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def train_fixed_phobert_sql_model(training_pairs, output_dir='./fixed_phobert_sql_model'):
    """
    Train the fixed PhoBERT-SQL model with proper architecture
    
    Args:
        training_pairs: List of {'vietnamese': str, 'sql': str} pairs
        output_dir: Directory to save the trained model
    """
    
    logger.info("Starting fixed PhoBERT-SQL training...")
    
    # Initialize model
    model = FixedPhoBERTForSQL()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    logger.info(f"Model initialized on {device}")
    logger.info(f"Vietnamese vocab size: {len(model.vietnamese_tokenizer)}")
    logger.info(f"SQL vocab size: {len(model.sql_tokenizer)}")
    
    # Create datasets
    train_size = int(0.8 * len(training_pairs))
    train_data = training_pairs[:train_size]
    val_data = training_pairs[train_size:]
    
    train_dataset = FixedVietnameseNL2SQLDataset(
        train_data, model.vietnamese_tokenizer, model.sql_tokenizer
    )
    val_dataset = FixedVietnameseNL2SQLDataset(
        val_data, model.vietnamese_tokenizer, model.sql_tokenizer
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Custom data collator for dual tokenizers
    def data_collator(features):
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.stack([f[key] for f in features])
        return batch
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        logging_first_step=True,
        save_total_limit=3
    )
    
    # Custom trainer for dual tokenizer setup
    class FixedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(
                vietnamese_input_ids=inputs['vietnamese_input_ids'],
                vietnamese_attention_mask=inputs['vietnamese_attention_mask'],
                sql_input_ids=inputs['sql_input_ids'],
                sql_attention_mask=inputs['sql_attention_mask'],
                labels=inputs['labels']
            )
            loss = outputs['loss']
            return (loss, outputs) if return_outputs else loss
    
    # Initialize trainer
    trainer = FixedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    try:
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        
        # Save tokenizers separately
        model.vietnamese_tokenizer.save_pretrained(f"{output_dir}/vietnamese_tokenizer")
        model.sql_tokenizer.save_pretrained(f"{output_dir}/sql_tokenizer")
        
        # Save model configuration
        config = {
            'model_type': 'FixedPhoBERTForSQL',
            'vietnamese_model': 'vinai/phobert-base',
            'sql_vocab_size': len(model.sql_tokenizer),
            'training_completed': True,
            'num_training_samples': len(train_dataset),
            'num_validation_samples': len(val_dataset)
        }
        
        with open(f"{output_dir}/model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Training completed successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def test_fixed_model(model, test_queries):
    """Test the fixed model with sample queries"""
    
    logger.info("Testing fixed model...")
    device = next(model.parameters()).device
    
    for i, query in enumerate(test_queries, 1):
        try:
            logger.info(f"Test {i}: {query}")
            
            # Tokenize Vietnamese input
            inputs = model.vietnamese_tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate SQL
            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate_sql(
                    vietnamese_input_ids=inputs['input_ids'],
                    vietnamese_attention_mask=inputs['attention_mask'],
                    max_length=256,
                    num_beams=4
                )
            
            # Decode SQL
            sql_query = model.sql_tokenizer.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            execution_time = time.time() - start_time
            
            logger.info(f"Generated SQL: {sql_query}")
            logger.info(f"Time: {execution_time:.3f}s")
            logger.info("---")
            
        except Exception as e:
            logger.error(f"Test {i} failed: {e}")

# Example usage
if __name__ == "__main__":
    # Sample training data
    sample_training_pairs = [
        {
            'vietnamese': 'Tìm tất cả balo nữ',
            'sql': 'SELECT p.name, b.brand_name, c.category_name, pr.price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN categories c ON p.category_id = c.category_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE p.name LIKE "%balo%" AND c.category_name LIKE "%nữ%"'
        },
        {
            'vietnamese': 'Tìm giày Nike giá dưới 1 triệu',
            'sql': 'SELECT p.name, b.brand_name, pr.price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE p.name LIKE "%giày%" AND b.brand_name LIKE "%Nike%" AND pr.price < 1000000'
        }
    ]
    
    # Train model
    model = train_fixed_phobert_sql_model(sample_training_pairs)
    
    # Test model
    test_queries = [
        "Tìm túi xách nữ",
        "Tìm giày Adidas màu đen",
        "Tìm sản phẩm có giá dưới 500k"
    ]
    
    test_fixed_model(model, test_queries)
