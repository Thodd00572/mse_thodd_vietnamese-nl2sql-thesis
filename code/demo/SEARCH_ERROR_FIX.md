# Search Network Error - Diagnosis & Fix

## 🐛 **Problem:**
When clicking "Search Products" button with a Vietnamese query, the application shows:
```
❌ API Error: Network Error
Search failed
```

Despite the Config page showing "API is connected" with green indicators.

---

## 🔍 **Root Cause Analysis:**

### **Issue 1: Backend Server Not Running**
The **backend API server on port 8000 is not running**.

**Evidence:**
- Frontend makes request to: `http://localhost:8000/api/search`
- Config page checks: `https://abnormally-direct-rhino.ngrok-free.app/health` (Colab)
- Config page shows "connected" because **it checks Colab, not local backend**
- When search button clicked → tries localhost:8000 → **connection refused**

### **Issue 2: Confusing Status Display**
The Config page shows "connected" status for Colab pipelines, but:
- ✅ Colab API is running (ngrok URL works)
- ❌ **Local backend is NOT running** (localhost:8000 not available)
- Frontend needs BOTH:
  1. Local backend to receive requests
  2. Colab API for pipeline execution

---

## ✅ **Solution:**

### **Step 1: Start the Backend Server**

**Option A: Using start_server.py (Recommended)**
```bash
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/backend/core
python start_server.py
```

This will start:
- Backend API on http://localhost:8000
- Frontend on http://localhost:3000 (if not already running)

**Option B: Using uvicorn directly**
```bash
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/backend/core
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

### **Step 2: Verify Backend is Running**

Open a new terminal and run:
```bash
curl http://localhost:8000/docs
```

**Expected:** HTML response from FastAPI docs page
**If Error:** Backend not running, repeat Step 1

---

### **Step 3: Test Search Again**

1. Go to http://localhost:3000
2. Enter query: "Hiển thị các giày thể thao nam có giá dưới 500k"
3. Click "Search Products"
4. **Expected:** Process log shows API call successful

---

## 📊 **System Architecture (What's Needed):**

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Browser   │─────▶│ Local Backend│─────▶│  Colab Pipelines │
│ localhost:  │      │ localhost:   │      │  ngrok URL:      │
│    3000     │      │    8000      │      │  abnormally...   │
└─────────────┘      └──────────────┘      └─────────────────┘
     ^                      │
     │                      │
     └──────────────────────┘
    Search response flows back
```

**All 3 components must be running:**
1. ✅ Frontend (Next.js on port 3000)
2. ❌ **Backend (FastAPI on port 8000)** ← **MISSING!**
3. ✅ Colab API (ngrok URL)

---

## 🔧 **Why Config Page Shows "Connected":**

The Config page directly checks Colab:
```javascript
const colabResponse = await fetch(
  'https://abnormally-direct-rhino.ngrok-free.app/health'
)
```

This succeeds because **Colab IS running**.

BUT the search button goes through localhost:8000 first:
```javascript
const response = await api.post('/api/search', {...})
// Goes to http://localhost:8000/api/search
```

This fails because **local backend is NOT running**.

---

## ✅ **Quick Verification Steps:**

### **1. Check if Backend is Running:**
```bash
lsof -i :8000
```

**Expected output if running:**
```
COMMAND   PID  USER
Python   1234  thoduong
```

**If empty:** Backend not running → Start it!

### **2. Check Backend API Docs:**
Open browser: http://localhost:8000/docs

**Expected:** FastAPI Swagger UI
**If "Connection Refused":** Backend not running

### **3. Test Search Endpoint:**
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Tìm áo thun", "pipeline": "all"}'
```

**Expected:** JSON response with SQL results
**If "Connection refused":** Backend not running

---

## 🚀 **Complete Startup Sequence:**

### **Terminal 1: Backend**
```bash
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/backend/core
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Wait for:** `Application startup complete` message

### **Terminal 2: Frontend (if not running)**
```bash
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/frontend
npm run dev
```

**Wait for:** `ready - started server on 0.0.0.0:3000`

### **Terminal 3: Colab (already running)**
- Run CELL 7+8 in colab_all_pipelines.py
- Ngrok URL: https://abnormally-direct-rhino.ngrok-free.app

---

## 📝 **Summary:**

| Component | Status | What It Does |
|-----------|--------|--------------|
| **Frontend** | ✅ Running (port 3000) | User interface |
| **Backend** | ❌ **NOT Running** | Coordinates requests |
| **Colab API** | ✅ Running (ngrok) | Executes pipelines |

**Problem:** Missing backend server
**Solution:** Start backend on port 8000
**Command:** `python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

---

**File:** `/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/SEARCH_ERROR_FIX.md`
**Created:** 2025-10-10
