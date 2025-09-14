# Google Colab Migration Guide - Vietnamese NL2SQL Project

## Benefits of Colab Migration

**Performance Gains:**
- **GPU Acceleration**: T4/V100 → 2-5 seconds inference (vs 15-20 seconds locally)
- **Memory**: 12-16GB GPU RAM vs local limitations
- **No Local Resources**: Save laptop CPU/memory

**Deployment Advantages:**
- **Easy Sharing**: Perfect for thesis demonstrations
- **Reproducible Environment**: Consistent across machines
- **Free GPU Access**: T4 GPU with free tier

## Migration Steps

### 1. Create Colab Notebook
```python
# Cell 1: Install Dependencies
!pip install fastapi uvicorn transformers torch
!pip install llama-cpp-python sentence-transformers
!pip install sqlite3 pandas numpy requests nest-asyncio

# Check GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 2. Upload Project Files
**Method A**: Drag & drop your `code/` and `data/` folders to Colab files panel
**Method B**: Create GitHub repo and clone:
```bash
!git clone https://github.com/yourusername/MSE_Thesis_2025.git
%cd MSE_Thesis_2025
```

### 3. GPU-Optimized Model Loading
```python
class ColabModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tokenizers = {}
    
    def load_sqlcoder_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "defog/sqlcoder-7b-2",
            torch_dtype=torch.float16,  # GPU optimization
            device_map="auto",
            low_cpu_mem_usage=True
        )
        return True
```

### 4. Database Setup
```python
# Create SQLite database in Colab
import sqlite3
conn = sqlite3.connect('/content/tiki_products.db')
# Load your CSV data here
```

### 5. Test Performance
```python
# Test query
result = pipeline2.full_pipeline("tìm điện thoại Samsung")
print(f"Execution time: {result['execution_time']:.2f}s")  # Should be 2-5s on GPU

2. **Pipeline 2**: Upload and run `Pipeline2_Colab.ipynb`
   - Loads Vietnamese-English translator and SQLCoder
   - Exposes API on port 8001 via ngrok
   - Copy the ngrok URL for configuration

### 3. Configure System

1. Open frontend at http://localhost:3001
2. Navigate to "Config" page
3. Paste Colab ngrok URLs
4. Save configuration
5. Verify green status indicators

### 4. Test End-to-End

1. Go to Search page
2. Try Vietnamese queries like:
   - "tìm iPhone giá rẻ"
   - "laptop Apple MacBook"
   - "tai nghe Sony chống ồn"
3. Compare performance between pipelines
4. Test fallback by stopping Colab notebooks

## API Endpoints

### Colab Configuration
- `POST /api/config/colab` - Configure Colab API URLs
- `GET /api/config/colab/status` - Check connection status

### Search with Hybrid Processing
- `POST /api/search` - Uses Colab APIs with local fallback
- `GET /api/health` - System health check

## Frontend Features

### Main Search Page
- Real-time Colab status indicators
- Warning banners for API unavailability
- Performance comparison between pipelines
- Automatic fallback messaging

### Configuration Page
- Easy Colab URL setup
- Connection health monitoring
- Architecture overview
- Performance benefits explanation

## Performance Comparison

| Mode | Pipeline 1 | Pipeline 2 | Notes |
|------|------------|------------|-------|
| **Colab GPU** | ~2-3 seconds | ~3-5 seconds | Optimal performance |
| **Local Fallback** | Instant (rule-based) | ~15-20 seconds | CPU inference |

## Troubleshooting

### Common Issues

1. **Port conflicts**: Backend uses 8080, frontend uses 3001
2. **Colab timeouts**: Restart notebooks if inactive
3. **ngrok limits**: Free tier has connection limits
4. **Model loading**: First Colab run takes longer

### Health Checks

- Frontend shows real-time status indicators
- Backend logs all API calls and fallbacks
- Configuration page displays connection health

## File Structure

```
MSE_Thesis_2025/
├── Pipeline1_Colab.ipynb          # Colab notebook for Pipeline 1
├── Pipeline2_Colab.ipynb          # Colab notebook for Pipeline 2
├── code/backend/
│   ├── app.py                     # Modified with Colab integration
│   ├── colab_client.py           # New Colab API client
│   └── models/pipelines.py       # Original pipeline implementations
└── code/frontend/
    ├── pages/
    │   ├── index.js              # Updated search page with status
    │   └── config.js             # New configuration page
    └── components/Layout.js       # Updated navigation
```

## Next Steps for Production

1. **Cloud Deployment**: Replace Colab with dedicated cloud VMs
2. **Authentication**: Add API keys for Colab endpoints
3. **Monitoring**: Implement comprehensive logging and metrics
4. **Scaling**: Add load balancing for multiple model instances
5. **Caching**: Cache frequent queries and model responses

## Thesis Demonstration

The system is now ready for thesis demonstration with:
- Live performance comparison between pipelines
- Real-time switching between cloud and local processing
- Visual status indicators and performance metrics
- Comprehensive fallback mechanisms

Access the system at:
- **Frontend**: http://localhost:3001
- **Backend**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
