# Vietnamese NL2SQL API Deployment

This directory contains the API server and evaluation scripts for the Vietnamese NL2SQL pipeline.

## Architecture

- **Training**: PhoBERT-SQL model trained in Google Colab with GPU acceleration
- **Deployment**: FastAPI server serves the trained model locally
- **Evaluation**: Local scripts call API endpoints and measure performance

## Files

- `api_server.py` - FastAPI server that serves the trained PhoBERT-SQL model
- `local_evaluation.py` - Comprehensive evaluation script with database generation
- `start_api.sh` - Script to start the API server
- `run_evaluation.sh` - Script to run evaluation
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Train Model (in Colab)

Run the Colab notebook to train the PhoBERT-SQL model:
```python
# In Colab: code/ColabNotebook/MSE_Thesis_Modular_Pipelines.py
trained_model = train_phobert_sql_model()
```

The model will be saved to Google Drive at:
- Model: `/content/drive/MyDrive/MSE_Thesis_Models/phobert_sql_trained.pth`
- Tokenizer: `/content/drive/MyDrive/MSE_Thesis_Models/phobert_sql_trained_tokenizer/`

### 2. Download Model Files

Download the trained model files from Google Drive to your local machine.

### 3. Start API Server

```bash
cd code/backend
./start_api.sh /path/to/phobert_sql_trained
```

The API will be available at `http://localhost:8000`

### 4. Run Evaluation

```bash
./run_evaluation.sh http://localhost:8000 100 evaluation_results.json
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Query Processing
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"vietnamese_query": "Tìm tất cả balo màu đen"}'
```

## Evaluation Features

- **Database Generation**: Creates sample Vietnamese e-commerce database
- **Query Complexity**: Tests simple, medium, and complex queries
- **Performance Metrics**: Measures API success rate, SQL execution rate, and latency
- **Comprehensive Results**: Saves detailed results to JSON file

## Sample Queries

- Simple: "Tìm tất cả balo", "Tìm giày thể thao"
- Medium: "Tìm balo màu đen giá dưới 1 triệu"
- Complex: "Tìm top 5 sản phẩm có đánh giá cao nhất"

## Expected Results

The evaluation will generate metrics including:
- API success rate
- Database execution rate  
- Average latency per query
- Performance breakdown by complexity level

## Troubleshooting

1. **Model not found**: Ensure model files are downloaded from Colab
2. **API not responding**: Check if server started successfully
3. **Import errors**: Install dependencies with `pip install -r requirements.txt`
4. **CUDA errors**: API will fallback to CPU if GPU not available
