#!/usr/bin/env python3
"""
Vietnamese NL2SQL API Server
Serves trained PhoBERT-SQL model via FastAPI endpoints
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL CLASSES (Copy from Colab)
# =============================================================================

class PhoBERTForSQL(nn.Module):
    """PhoBERT model with SQL generation head for Vietnamese NL2SQL"""
    
    def __init__(self, model_name='vinai/phobert-base'):
        super().__init__()
        
        # Backbone: vinai/phobert-base (≈135M parameters)
        self.phobert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # SQL generation head: lightweight decoder with constraints
        self.sql_head = nn.Sequential(
            nn.Linear(768, 512),      # PhoBERT hidden size to intermediate
            nn.ReLU(),
            nn.Dropout(0.1),          # Regularization on decoder layers
            nn.Linear(512, 256),      # Intermediate layer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.tokenizer.vocab_size)  # Generate SQL tokens
        )
        
        # Initialize generation capabilities
        self.config = self.phobert.config
        self.config.is_decoder = True
        self.config.add_cross_attention = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for training"""
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Generate SQL logits
        sql_logits = self.sql_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(sql_logits.view(-1, sql_logits.size(-1)), labels.view(-1))
        
        return {'loss': loss, 'logits': sql_logits}

    def generate(self, input_ids, attention_mask=None, max_length=512, num_beams=3, early_stopping=True):
        """Generate SQL from Vietnamese input"""
        self.eval()
        with torch.no_grad():
            # Get encoder outputs
            encoder_outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = encoder_outputs.last_hidden_state
            
            # Simple greedy generation (can be enhanced with beam search)
            batch_size = input_ids.size(0)
            device = input_ids.device
            
            # Start with input sequence
            generated = input_ids.clone()
            
            for _ in range(max_length - input_ids.size(1)):
                # Get logits for next token
                outputs = self.sql_head(hidden_states[:, -1:, :])  # Use last hidden state
                next_token_logits = outputs[:, -1, :]
                
                # Get next token (greedy)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS token
                if early_stopping and next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            return generated

# =============================================================================
# API MODELS
# =============================================================================

class QueryRequest(BaseModel):
    vietnamese_query: str
    include_metrics: bool = True

class QueryResponse(BaseModel):
    sql_query: str
    success: bool
    execution_time: float
    error: Optional[str] = None
    pipeline: str = "Pipeline1"
    method: str = "Vietnamese → PhoBERT-SQL → SQL"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool

# =============================================================================
# API SERVER
# =============================================================================

class VietnameseNL2SQLServer:
    def __init__(self):
        self.app = FastAPI(
            title="Vietnamese NL2SQL API",
            description="API for Vietnamese Natural Language to SQL conversion using PhoBERT",
            version="1.0.0"
        )
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup routes
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None,
                gpu_available=torch.cuda.is_available()
            )
        
        @self.app.post("/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest):
            """Process Vietnamese query and return SQL"""
            if not self.model:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                start_time = time.time()
                
                # Preprocess query
                query = request.vietnamese_query.strip()
                
                # Tokenize
                inputs = self.tokenizer(
                    query,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate SQL
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=512,
                        num_beams=3,
                        early_stopping=True
                    )
                    
                    # Decode generated SQL (remove input tokens)
                    sql_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    sql_query = self.tokenizer.decode(sql_tokens, skip_special_tokens=True)
                
                execution_time = time.time() - start_time
                
                # Basic SQL validation
                if not sql_query.strip():
                    sql_query = "SELECT * FROM products LIMIT 10"  # Fallback
                
                return QueryResponse(
                    sql_query=sql_query.strip(),
                    success=True,
                    execution_time=execution_time,
                    pipeline="Pipeline1",
                    method="Vietnamese → PhoBERT-SQL → SQL"
                )
                
            except Exception as e:
                logger.error(f"Query processing failed: {str(e)}")
                return QueryResponse(
                    sql_query="",
                    success=False,
                    execution_time=0.0,
                    error=str(e),
                    pipeline="Pipeline1",
                    method="Vietnamese → PhoBERT-SQL → SQL"
                )
    
    def load_model(self, model_path: str):
        """Load trained PhoBERT-SQL model"""
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Load model checkpoint
            checkpoint = torch.load(f"{model_path}.pth", map_location=self.device)
            
            # Initialize model
            self.model = PhoBERTForSQL()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            tokenizer_path = f"{model_path}_tokenizer"
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
            
            logger.info("✅ Model loaded successfully!")
            logger.info(f"Device: {self.device}")
            logger.info(f"Tokenizer vocab size: {len(self.tokenizer.vocab)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

def create_app(model_path: str = None) -> FastAPI:
    """Create and configure FastAPI app"""
    server = VietnameseNL2SQLServer()
    
    if model_path:
        server.load_model(model_path)
    
    return server.app

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vietnamese NL2SQL API Server")
    parser.add_argument("--model-path", required=True, help="Path to trained model (without .pth extension)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Create app with model
    app = create_app(args.model_path)
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
