# Vietnamese-to-SQL Translation Thesis Experiment

## Overview
Full-stack experimental application comparing two Vietnamese-to-SQL translation pipelines for MSE thesis research.

## Architecture
- **Backend**: FastAPI + Hugging Face Transformers + PyTorch
- **Frontend**: React/Next.js with Tailwind CSS
- **Database**: SQLite with Tiki product dataset
- **Models**: PhoBERT-SQL, PhoBERT translation, SQLCoder

## Project Structure
```
code/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main application
│   ├── start_server.py     # Startup script
│   ├── config.py           # Configuration
│   ├── api/                # API routes
│   ├── models/             # ML model management
│   ├── database/           # Database management
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── pages/              # React pages
│   ├── components/         # Reusable components
│   ├── utils/              # API utilities
│   ├── styles/             # CSS styles
│   └── package.json        # Node dependencies
└── start_thesis_app.sh     # Complete startup script
```

## Quick Start
1. **Start the complete application:**
   ```bash
   cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code
   ./start_thesis_app.sh
   ```

2. **Manual startup (alternative):**
   ```bash
   # Backend
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python start_server.py
   
   # Frontend (new terminal)
   cd frontend
   npm install
   npm run dev
   ```

## Pipelines
1. **Pipeline 1**: Vietnamese → PhoBERT-SQL → SQL → Execute
2. **Pipeline 2**: Vietnamese → PhoBERT → English → SQLCoder → SQL → Execute

## API Endpoints
- `POST /api/search` - Execute translation pipelines
- `GET /api/analyze` - Get performance metrics
- `GET /api/export/{type}` - Export data (CSV/JSON)
- `GET /api/schema` - Database schema info
- `GET /api/health` - System health check

## Frontend Pages
- **Search** (`/`) - Vietnamese query input with side-by-side pipeline results
- **Analysis** (`/analysis`) - Performance metrics and comparison charts
- **Database** (`/database`) - Schema explorer and SQL query executor

## Research Features
- **Metrics Tracking**: EX (Execution Accuracy), EM (Exact Match), latency, GPU memory
- **CSV Export**: All metrics and results for thesis analysis
- **System Monitoring**: CPU, memory, GPU utilization tracking
- **Comparative Analysis**: Side-by-side pipeline performance

## Sample Vietnamese Queries
- "tìm iPhone giá rẻ"
- "laptop Apple MacBook"
- "tai nghe Sony chống ồn"
- "điện thoại Samsung Galaxy"
- "sản phẩm có đánh giá cao"

## Development Notes
- Models use placeholder implementations initially
- Replace with actual PhoBERT-SQL models when available
- SQLite database includes realistic Tiki product data
- All metrics are tracked for thesis analysis
