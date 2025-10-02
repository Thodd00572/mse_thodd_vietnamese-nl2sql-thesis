import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel, MarianTokenizer, MarianMTModel
import logging

logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration for all Hugging Face models"""
    
    # Model identifiers
    PHOBERT_BASE = "vinai/phobert-base"
    PHOBERT_LARGE = "vinai/phobert-large"
    SQLCODER = "defog/sqlcoder-7b-2"
    
    # For Vietnamese-SQL models, we'll use available alternatives
    # Note: Replace these with actual PhoBERT-SQL models when available
    VN_SQL_MODEL = "microsoft/DialoGPT-medium"  # Placeholder
    VN_EN_TRANSLATE = "Helsinki-NLP/opus-mt-vi-en"  # Vietnamese to English
    
    @staticmethod
    def get_device():
        """Get the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

class ModelLoader:
    """Centralized model loader for all AI models"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = ModelConfig.get_device()
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"ModelLoader initialized with device: {self.device}")
        logger.info(f"Cache directory: {self.cache_dir}")
        
    def load_phobert_sql_model(self):
        """Load PhoBERT-SQL model (placeholder implementation)"""
        try:
            logger.info("Loading PhoBERT-SQL model...")
            # Using PhoBERT base as placeholder - replace with actual PhoBERT-SQL model
            tokenizer = AutoTokenizer.from_pretrained(ModelConfig.PHOBERT_BASE)
            model = AutoModel.from_pretrained(ModelConfig.PHOBERT_BASE)
            
            model.to(self.device)
            
            self.tokenizers['phobert_sql'] = tokenizer
            self.models['phobert_sql'] = model
            
            logger.info("PhoBERT-SQL model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading PhoBERT-SQL model: {e}")
            return False
    
    def load_vietnamese_english_translator(self):
        """Load Vietnamese to English translation model"""
        logger.info("[ModelLoader] Starting Vietnamese-English translator loading...")
        
        if 'vn_en_translate' in self.models:
            logger.info("[ModelLoader] Vietnamese-English translator already loaded")
            return True
            
        try:
            model_name = "Helsinki-NLP/opus-mt-vi-en"
            logger.info(f"[ModelLoader] Loading tokenizer: {model_name}")
            tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            logger.info("[ModelLoader] Vietnamese-English tokenizer loaded")
            
            logger.info(f"[ModelLoader] Loading model: {model_name}")
            model = MarianMTModel.from_pretrained(model_name, cache_dir=self.cache_dir)
            logger.info("[ModelLoader] Vietnamese-English model loaded")
            
            logger.info(f"[ModelLoader] Moving model to device: {self.device}")
            model = model.to(self.device)
            
            self.models['vn_en_translate'] = model
            self.tokenizers['vn_en_translate'] = tokenizer
            
            logger.info("[ModelLoader] Vietnamese-English translation model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"[ModelLoader] Failed to load Vietnamese-English translation model: {e}")
            return False
    
    def load_sqlcoder_model(self):
        """Load SQLCoder model using llama-cpp-python with transformers fallback"""
        logger.info("[ModelLoader] Starting SQLCoder model loading...")
        
        if 'sqlcoder' in self.models:
            logger.info("[ModelLoader] SQLCoder model already loaded")
            return True
            
        try:
            # Try to load with llama-cpp-python first
            logger.info("[ModelLoader] Attempting llama-cpp-python loading...")
            from llama_cpp import Llama
            
            # Use lightweight SQLCoder-7B GGUF (proven working)
            logger.info("[ModelLoader] Loading lightweight SQLCoder-7B GGUF...")
            model = Llama.from_pretrained(
                repo_id="defog/sqlcoder-7b-2",
                filename="sqlcoder-7b-q5_k_m.gguf",  # 5-bit quantized, good balance
                n_ctx=2048,
                n_threads=8,  # Optimize for local CPU
                n_gpu_layers=0,  # Keep on CPU for compatibility
                verbose=False
            )
            logger.info("[ModelLoader] Lightweight SQLCoder-7B loaded successfully")
            
            self.models['sqlcoder'] = model
            logger.info("[ModelLoader] SQLCoder model loaded successfully with llama-cpp-python")
            return True
            
        except Exception as e:
            logger.warning(f"[ModelLoader] llama-cpp-python loading failed: {e}")
            logger.info("[ModelLoader] Attempting transformers fallback...")
            
            try:
                # Fallback to transformers
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                # Use lightweight SQLCoder-7B only
                logger.info("[ModelLoader] Loading lightweight SQLCoder-7B with transformers...")
                tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2", cache_dir=self.cache_dir)
                
                # Load model with optimizations for lightweight deployment
                model = AutoModelForCausalLM.from_pretrained(
                    "defog/sqlcoder-7b-2",
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    device_map="auto" if self.device.type == 'cuda' else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                # Handle MPS device placement
                if self.device.type == 'mps':
                    model = model.to('cpu')
                    logger.info("[ModelLoader] SQLCoder-7B kept on CPU for MPS compatibility")
                else:
                    model = model.to(self.device)
                    logger.info(f"[ModelLoader] SQLCoder-7B moved to device: {self.device}")
                
                self.models['sqlcoder'] = model
                self.tokenizers['sqlcoder'] = tokenizer
                logger.info("[ModelLoader] Lightweight SQLCoder-7B loaded successfully with transformers")
                return True
                
            except Exception as e2:
                logger.error(f"[ModelLoader] Both llama-cpp-python and transformers loading failed: {e2}")
                return False
    
    def load_all_models(self):
        """Load all required models"""
        results = {
            'phobert_sql': self.load_phobert_sql_model(),
            'vn_en_translate': self.load_vietnamese_english_translator(),
            'sqlcoder': self.load_sqlcoder_model()
        }
        
        success_count = sum(results.values())
        logger.info(f"Loaded {success_count}/3 models successfully")
        
        return results
    
    def get_model(self, model_name: str):
        """Get a loaded model by name"""
        model = self.models.get(model_name)
        if model:
            logger.debug(f"[ModelLoader] Retrieved model '{model_name}': {type(model)}")
        else:
            logger.warning(f"[ModelLoader] Model '{model_name}' not found in cache")
        return model
    
    def get_tokenizer(self, model_name: str):
        """Get a loaded tokenizer by name"""
        tokenizer = self.tokenizers.get(model_name)
        if tokenizer:
            logger.debug(f"[ModelLoader] Retrieved tokenizer '{model_name}': {type(tokenizer)}")
        else:
            logger.debug(f"[ModelLoader] Tokenizer '{model_name}' not found in cache (may be None for llama-cpp models)")
        return tokenizer
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }
        else:
            return {"message": "CUDA not available"}

# Global model loader instance
model_loader = ModelLoader()
