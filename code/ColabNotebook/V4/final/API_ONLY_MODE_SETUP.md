# API-Only Mode Setup Guide

## 🎯 **What Was Changed:**

All evaluation processes have been **commented out** to enable **API-only deployment** without running time-consuming evaluations on `eval_data.jsonl`.

---

## ✅ **Changes Applied:**

### **1. P1 (mT5) Evaluation - COMMENTED OUT**

**Location:** Line 661-681

**Before:**
```python
if eval_data:
    metrics_p1, results_p1, pipeline_p1 = run_evaluation()
else:
    print("Upload eval_data.jsonl to Google Drive to run evaluation")
    pipeline_p1 = None
```

**After:**
```python
# ============================================================================
# EVALUATION COMMENTED OUT - API ONLY MODE
# ============================================================================
# Uncomment the block below to run P1 evaluation:
"""
if eval_data:
    metrics_p1, results_p1, pipeline_p1 = run_evaluation()
else:
    print("Upload eval_data.jsonl to Google Drive to run evaluation")
    pipeline_p1 = None
"""

# For API-only mode, set these to None
metrics_p1 = None
results_p1 = None
pipeline_p1 = None

print("\n" + "=" * 60)
print("P1 EVALUATION SKIPPED - API ONLY MODE")
print("Pipeline will be initialized automatically in CELL 7+8")
print("=" * 60)
```

---

### **2. P2 (SQLCoder) Evaluation - COMMENTED OUT**

**Location:** Line 2042-2056

**Before:**
```python
metrics_p2, results_p2, pipeline_p2 = main()
```

**After:**
```python
# ============================================================================
# EVALUATION COMMENTED OUT - API ONLY MODE
# ============================================================================
# Uncomment the line below to run P2 evaluation:
# metrics_p2, results_p2, pipeline_p2 = main()

# For API-only mode, set these to None
metrics_p2 = None
results_p2 = None
pipeline_p2 = None

print("\n" + "=" * 60)
print("P2 EVALUATION SKIPPED - API ONLY MODE")
print("Pipeline will be initialized automatically in CELL 7+8")
print("=" * 60)
```

---

### **3. P3 (Vanna AI) Evaluation - COMMENTED OUT**

**Location:** Line 4183-4209

**Before:**
```python
if eval_data:
    metrics_p3, results_p3, pipeline_p3 = run_evaluation()
else:
    print("Upload eval_data.jsonl to Google Drive to run evaluation")
    metrics_p3 = None
    results_p3 = None
    pipeline_p3 = None
```

**After:**
```python
# ============================================================================
# EVALUATION COMMENTED OUT - API ONLY MODE
# ============================================================================
# Uncomment the block below to run P3 evaluation:
"""
if eval_data:
    metrics_p3, results_p3, pipeline_p3 = run_evaluation()
else:
    print("Upload eval_data.jsonl to Google Drive to run evaluation")
    metrics_p3 = None
    results_p3 = None
    pipeline_p3 = None
"""

# For API-only mode, set these to None
metrics_p3 = None
results_p3 = None
pipeline_p3 = None

print("\n" + "=" * 60)
print("P3 EVALUATION SKIPPED - API ONLY MODE")
print("Pipeline will be initialized automatically in CELL 7+8")
print("=" * 60)
```

---

## 🚀 **How to Use API-Only Mode:**

### **Step 1: Run Required Setup Cells**

You only need to run these cells in Colab:

```
📋 CELL 1: Package Installation (run once)
📋 CELL 2: Google Drive Setup & Data Loading
📋 CELL 3: Evaluation Metrics & Utilities (optional, for utils)
```

**Skip these cells completely:**
- ❌ CELL 4: Pipeline definitions (will be auto-loaded)
- ❌ CELL 5: Evaluation functions (not needed for API)
- ❌ CELL 6: Main execution (evaluation commented out)

### **Step 2: Run API Setup Cells**

```
✅ CELL 7: FastAPI Setup for All Pipelines
✅ CELL 8: Start ngrok Tunnel and FastAPI Server
```

**CELL 7** will automatically:
- Check if pipeline classes exist
- Define them inline if missing
- Initialize all 3 pipelines (P1, P2, P3)
- Load models automatically
- Show clear status for each pipeline

**CELL 8** will:
- Start FastAPI server with uvicorn
- Open ngrok tunnel
- Expose all 3 pipeline endpoints
- Display public URL for API access

---

## 📊 **What You'll See:**

### **When Running P1/P2/P3 Sections:**

```
============================================================
P1 EVALUATION SKIPPED - API ONLY MODE
Pipeline will be initialized automatically in CELL 7+8
============================================================
```

```
============================================================
P2 EVALUATION SKIPPED - API ONLY MODE
Pipeline will be initialized automatically in CELL 7+8
============================================================
```

```
============================================================
P3 EVALUATION SKIPPED - API ONLY MODE
Pipeline will be initialized automatically in CELL 7+8
============================================================
```

### **When Running CELL 7:**

```
============================================================
Setting up Combined FastAPI for ALL Pipelines
P1: mT5 Zero-Shot | P2: SQLCoder Zero-Shot | P3: Vanna AI RAG
============================================================

Checking/Installing API dependencies...
[OK] All API packages already installed

[INFO] Pipeline classes not found. Defining them for API use...
[OK] Pipeline classes defined for API use

============================================================
Checking and Initializing Pipelines for API...
============================================================

[INFO] P1 (mT5) not found in memory. Initializing for API...
Loading google/mt5-base...
Model loaded on cuda
[✓] P1 (mT5 Zero-Shot) initialized successfully!

[INFO] P2 (SQLCoder) not found in memory. Initializing for API...
Loading SQLCoder model: defog/sqlcoder-7b-2
⏳ This may take several minutes for first-time download (~14GB)...
✅ Model loaded with 8-bit quantization
✅ SQLCoder ready on cuda
[✓] P2 (SQLCoder Zero-Shot) initialized successfully!

[INFO] P3 (Vanna AI) not found in memory. Initializing for API...
[✓] P3 (Vanna AI RAG) initialized successfully!

============================================================
Pipeline Initialization Summary:
============================================================
  P1 (mT5):       ✅ READY
  P2 (SQLCoder):  ✅ READY
  P3 (Vanna AI):  ✅ READY
============================================================
```

### **When Running CELL 8:**

```
🌐 Starting ngrok tunnel...
✅ ngrok tunnel established!

📡 PUBLIC URL: https://abnormally-direct-rhino.ngrok-free.app

============================================================
API ENDPOINTS:
============================================================

Root:
  /                    (GET)  - API information
  /health              (GET)  - Health check
  /config/colab/status (GET)  - Frontend config

Pipeline 1 (mT5):
  /p1/generate         (POST) - Generate SQL
  /p1/metrics          (GET)  - Get metrics

Pipeline 2 (SQLCoder):
  /p2/generate         (POST) - Generate SQL
  /p2/metrics          (GET)  - Get metrics

Pipeline 3 (Vanna AI):
  /p3/generate         (POST) - Generate SQL
  /p3/metrics          (GET)  - Get metrics

============================================================
🚀 API SERVER RUNNING!
Access your API at: https://abnormally-direct-rhino.ngrok-free.app
============================================================
```

---

## 🔌 **Testing API Endpoints:**

### **1. Health Check**

```bash
curl https://abnormally-direct-rhino.ngrok-free.app/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0",
  "pipelines": {
    "P1": true,
    "P2": true,
    "P3": true
  },
  "device": "cuda"
}
```

---

### **2. Test P1 (mT5)**

```bash
curl -X POST "https://abnormally-direct-rhino.ngrok-free.app/p1/generate" \
     -H "Content-Type: application/json" \
     -d '{"query": "Tìm áo thun"}'
```

**Response:**
```json
{
  "pipeline": "P1_mT5_Zero_Shot",
  "sql_query": "SELECT * FROM products WHERE name LIKE '%áo thun%' LIMIT 10;",
  "execution_time": 0.234,
  "valid": true,
  "success": true,
  "metrics": {
    "latency_ms": 234,
    "model": "google/mt5-base"
  }
}
```

---

### **3. Test P2 (SQLCoder)**

```bash
curl -X POST "https://abnormally-direct-rhino.ngrok-free.app/p2/generate" \
     -H "Content-Type: application/json" \
     -d '{"query": "Hiển thị sản phẩm có giá dưới 500000"}'
```

**Response:**
```json
{
  "pipeline": "P2_SQLCoder_Zero_Shot",
  "sql_query": "SELECT * FROM products WHERE current_price < 500000 LIMIT 10;",
  "execution_time": 1.456,
  "valid": true,
  "success": true,
  "metrics": {
    "latency_ms": 1456,
    "model": "defog/sqlcoder-7b-2"
  }
}
```

---

### **4. Test P3 (Vanna AI)**

```bash
curl -X POST "https://abnormally-direct-rhino.ngrok-free.app/p3/generate" \
     -H "Content-Type: application/json" \
     -d '{"query": "Top 10 sản phẩm bán chạy nhất"}'
```

**Response:**
```json
{
  "pipeline": "P3_Vanna_AI_RAG",
  "sql_query": "SELECT p.name, b.brand_name, pr.current_price, rv.rating_average, pr.quantity_sold FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id JOIN product_reviews rv ON p.product_id = rv.product_id ORDER BY pr.quantity_sold DESC, rv.rating_average DESC LIMIT 10;",
  "execution_time": 2.123,
  "valid": true,
  "success": true,
  "metrics": {
    "latency_ms": 2123,
    "model": "gpt-4o-mini + ChromaDB"
  }
}
```

---

## 🌐 **Frontend Integration:**

Your local frontend (`http://localhost:3000`) can now call these APIs:

```javascript
// Example: Search with all pipelines
const response = await fetch('https://abnormally-direct-rhino.ngrok-free.app/p1/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'Tìm giày thể thao'
  })
});

const result = await response.json();
console.log(result.sql_query);
```

---

## ⏱️ **Time Savings:**

### **With Evaluation (Old Way):**
```
P1 Evaluation: ~15 minutes (300 queries)
P2 Evaluation: ~45 minutes (300 queries, large model)
P3 Evaluation: ~30 minutes (300 queries, API calls)
────────────────────────────────────────────────
Total: ~90 minutes before API available
```

### **API-Only Mode (New Way):**
```
CELL 7 Setup: ~2-3 minutes (model downloads)
CELL 8 Start:  ~10 seconds (server start)
────────────────────────────────────────────────
Total: ~3 minutes to API availability! 🚀
```

**Time Saved: ~87 minutes (97% faster!)**

---

## 🔄 **If You Need Evaluation Later:**

### **To Re-enable P1 Evaluation:**

1. Go to line 664
2. Uncomment the evaluation block:

```python
# Remove the triple quotes:
if eval_data:
    metrics_p1, results_p1, pipeline_p1 = run_evaluation()
else:
    print("Upload eval_data.jsonl to Google Drive to run evaluation")
    pipeline_p1 = None

# Comment out the API-only lines:
# metrics_p1 = None
# results_p1 = None
# pipeline_p1 = None
```

### **To Re-enable P2 Evaluation:**

1. Go to line 2046
2. Uncomment:

```python
metrics_p2, results_p2, pipeline_p2 = main()
```

3. Comment out:

```python
# metrics_p2 = None
# results_p2 = None
# pipeline_p2 = None
```

### **To Re-enable P3 Evaluation:**

1. Go to line 4187
2. Remove the triple quotes around the evaluation block
3. Comment out the API-only lines

---

## 📋 **Quick Start Checklist:**

- [ ] Run CELL 1: Package installation
- [ ] Run CELL 2: Google Drive setup
- [ ] Skip P1/P2/P3 evaluation sections (already commented out)
- [ ] Run CELL 7: FastAPI setup
- [ ] Wait for all 3 pipelines to show ✅ READY
- [ ] Run CELL 8: Start server
- [ ] Copy ngrok public URL
- [ ] Test API endpoints with curl or from frontend
- [ ] Verify all 3 pipelines work

---

## ⚠️ **Important Notes:**

### **1. No Metrics Available**

Since evaluations are skipped:
- `/p1/metrics` → 404 (no metrics)
- `/p2/metrics` → 404 (no metrics)  
- `/p3/metrics` → 404 (no metrics)

**Only generation endpoints work:**
- `/p1/generate` ✅
- `/p2/generate` ✅
- `/p3/generate` ✅

### **2. Model Download Times**

**First Run:**
- P1 (mT5): ~1 min (~500 MB)
- P2 (SQLCoder): ~3-5 min (~14 GB)
- P3 (Vanna AI): ~30 sec (uses OpenAI API)

**Subsequent Runs:**
- Models are cached
- Load time: ~30-60 seconds total

### **3. GPU Memory Requirements**

| Pipeline | Memory | GPU |
|----------|--------|-----|
| P1 (mT5) | ~2.5 GB | T4+ |
| P2 (SQLCoder 8-bit) | ~7 GB | T4+ |
| P3 (Vanna AI) | Minimal | Any |
| **Total (all 3)** | ~10 GB | **T4 (16GB) OK** |

### **4. API Key for P3**

Vanna AI (P3) requires OpenAI API key:
- Set in Colab Secrets: `OPENAI_API_KEY`
- Or set `MANUAL_API_KEY` variable at top of notebook
- Without key, P3 will show NOT LOADED (P1 and P2 still work)

---

## 🎉 **Summary:**

| Feature | Before | After |
|---------|--------|-------|
| **Time to API** | ~90 min | ~3 min |
| **Evaluation runs** | ✅ Required | ❌ Skipped |
| **API availability** | After eval | Immediate |
| **All pipelines exposed** | ✅ | ✅ |
| **Frontend compatible** | ✅ | ✅ |
| **Can re-enable eval** | N/A | ✅ |

**Result:** Fast API deployment without waiting for evaluations! 🚀

---

**File Modified:** `/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/ColabNotebook/V4/final/colab_all_pipelines.py`

**Lines Changed:**
- P1: Lines 661-681
- P2: Lines 2042-2056
- P3: Lines 4183-4209

**Status:** ✅ **COMPLETED** - API-Only Mode Ready!

---

**Date:** 2025-10-10  
**Mode:** API-Only Deployment  
**Purpose:** Fast API exposure without evaluation overhead
