# Vietnamese NL2SQL Demo Application

**Full-stack demonstration application for Vietnamese NL2SQL thesis**

This is a working demo application that showcases the Vietnamese NL2SQL pipelines with a web interface for live demonstrations and presentations.

---

## 🎯 Purpose

- **Demonstrations**: Live demo during thesis defense or presentations
- **Experimentation**: Interactive testing of Vietnamese queries
- **Visualization**: Side-by-side pipeline comparison with UI
- **Analysis**: Real-time metrics and performance charts

---

## 📁 Structure

```
demo/
├── backend/                 # FastAPI backend server
│   ├── api/                # REST API endpoints
│   ├── models/             # Model serving infrastructure
│   ├── database/           # Database management
│   ├── evaluation/         # Evaluation system
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # React/Next.js frontend
│   ├── pages/             # React pages
│   ├── components/        # UI components
│   ├── utils/             # API utilities
│   └── package.json       # Node dependencies
│
└── start_thesis_app.sh    # Complete startup script
```

---

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)

```bash
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo
./start_thesis_app.sh
```

This will:
- ✅ Create Python virtual environment
- ✅ Install all dependencies
- ✅ Start backend on `http://localhost:8000`
- ✅ Start frontend on `http://localhost:3000`

### Option 2: Manual Startup

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python start_server.py
```

**Frontend (new terminal):**
```bash
cd frontend
npm install
npm run dev
```

---

## 🌐 Access Points

Once running, access:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)

---

## 📊 Features

### Frontend Pages

1. **Search** (`/`)
   - Vietnamese query input
   - Side-by-side pipeline results
   - Execution time comparison
   - SQL query display

2. **Analysis** (`/analysis`)
   - Performance metrics charts
   - EM/EX comparison graphs
   - Latency analysis
   - Success rate tracking

3. **Database** (`/database`)
   - Schema explorer
   - Direct SQL query executor
   - Table browser

### Backend API

- `POST /api/search` - Execute Vietnamese query through pipelines
- `GET /api/analyze` - Get performance metrics
- `GET /api/export/{type}` - Export data (CSV/JSON)
- `GET /api/schema` - Database schema information
- `GET /api/health` - System health check

---

## 🎓 Use Cases

### Thesis Defense Demonstration
```
1. Open http://localhost:3000
2. Type Vietnamese query: "Tìm iPhone giá rẻ"
3. Show side-by-side results from both pipelines
4. Explain performance differences
5. Navigate to /analysis for metrics
```

### Live Testing
```
1. Go to Search page
2. Test various query complexities
3. Show real-time SQL generation
4. Demonstrate execution results
```

### Performance Analysis
```
1. Open /analysis page
2. Show EM/EX metrics
3. Display latency comparisons
4. Export results for thesis
```

---

## ⚙️ Configuration

### Backend Models

The backend currently uses **placeholder models** for demonstration:
- Pipeline 1: Vietnamese → SQL direct generation
- Pipeline 2: Vietnamese → English → SQL translation

To use **production models**:
1. Train models in Google Colab (see `/ColabNotebook/V4/final/`)
2. Download trained models
3. Update `backend/models/` with actual model files
4. Configure paths in `backend/config.py`

### Database

The demo uses the same Tiki database:
- Location: `/ColabNotebook/db/tiki.sqlite`
- Copy to: `backend/database/tiki.sqlite`
- Or update path in configuration

---

## 🔧 Technical Stack

### Backend
- **Framework**: FastAPI (Python)
- **Models**: Hugging Face Transformers + PyTorch
- **Database**: SQLite
- **API**: REST with automatic OpenAPI docs

### Frontend
- **Framework**: Next.js (React)
- **Styling**: Tailwind CSS
- **Charts**: Recharts for visualization
- **State**: React hooks

---

## 📝 Sample Queries for Demo

**Simple:**
- "tìm iPhone giá rẻ"
- "laptop Apple MacBook"
- "tai nghe Sony chống ồn"

**Medium:**
- "điện thoại Samsung Galaxy dưới 10 triệu"
- "sản phẩm có đánh giá cao nhất"
- "top 10 laptop bán chạy"

**Complex:**
- "so sánh giá iPhone và Samsung"
- "phân tích xu hướng giá laptop"
- "thống kê sản phẩm theo thương hiệu"

---

## 🛑 Stopping the Application

If started with `start_thesis_app.sh`:
```bash
# The script prints PIDs on startup, use:
kill <BACKEND_PID> <FRONTEND_PID>
```

Or manually:
```bash
# Find processes
ps aux | grep "uvicorn\|next"

# Kill them
pkill -f uvicorn
pkill -f next
```

---

## 🔍 Troubleshooting

**Backend won't start:**
- Check Python version (3.8+)
- Verify all requirements installed
- Check port 8000 is free

**Frontend won't start:**
- Check Node.js version (14+)
- Run `npm install` again
- Check port 3000 is free

**Models not loading:**
- This is expected with placeholder models
- For production: download trained models from Colab

---

## 📊 Relationship to Production Code

| Component | Purpose | Production Code |
|-----------|---------|----------------|
| **Demo App** | Demonstrations, UI | This directory |
| **Production Pipelines** | Actual evaluation | `/ColabNotebook/V4/final/` |
| **Evaluation Data** | Test queries | `/ColabNotebook/data/` |
| **Database** | Tiki data | `/ColabNotebook/db/` |

**Important**: 
- Demo app uses **placeholder models** for UI demonstration
- **Real evaluation** happens in Colab notebooks
- **Production metrics** (76.3% EX) from Colab, not demo app

---

## ✅ Status

- ✅ **Frontend**: Fully functional UI
- ✅ **Backend**: API endpoints working
- ⚠️ **Models**: Placeholder (for demo only)
- ✅ **Database**: Using actual Tiki data
- ✅ **Metrics**: Tracking system implemented

---

## 🎯 For Thesis Defense

**Recommended Demo Flow:**
1. Show Search interface with simple query
2. Display both pipeline results side-by-side
3. Navigate to Analysis page for metrics
4. Show Database schema explorer
5. Explain architecture using API docs
6. Reference production results from Colab

**Key Message**: 
"This demo shows the *concept* with placeholder models. The actual *production results* (76.3% EX) come from the Colab notebooks with real trained models."

---

**Last Updated**: October 1, 2025  
**Status**: ✅ Ready for demonstrations  
**Production Code**: See `/ColabNotebook/V4/final/` for actual evaluation  
