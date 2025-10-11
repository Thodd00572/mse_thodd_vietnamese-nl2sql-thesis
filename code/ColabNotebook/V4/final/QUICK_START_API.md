# Quick Start: API-Only Deployment

## 🚀 **3-Minute Setup for API Exposure**

---

## ✅ **What to Run:**

### **Step 1: Basic Setup (Optional)**

```
CELL 1: Package Installation (run once if packages missing)
CELL 2: Google Drive Setup (optional, for database access)
```

**Note:** These cells are optional if you just want API endpoints without database execution.

---

### **Step 2: API Setup (Required)**

```
CELL 7: FastAPI Setup for All Pipelines  ⭐ START HERE
CELL 8: Start ngrok Tunnel and Server     ⭐ THEN THIS
```

**That's it!** Just 2 cells to expose all 3 pipelines via API!

---

## ❌ **What to Skip:**

```
❌ CELL 3: Evaluation Metrics (not needed for API)
❌ CELL 4: Pipeline definitions (auto-loaded in CELL 7)
❌ CELL 5: Evaluation functions (commented out)
❌ CELL 6: Main execution (commented out)
```

**All evaluation code is already commented out!**

---

## 📋 **Execution Flow:**

```
┌──────────────────────────────────────────┐
│  Open Colab Notebook                     │
│  colab_all_pipelines.py                  │
└──────────────────────────────────────────┘
                 │
                 ↓
┌──────────────────────────────────────────┐
│  Run CELL 7: FastAPI Setup               │
│  ⏱️  Wait 2-3 minutes                     │
│                                          │
│  Expected output:                        │
│  ✅ P1 (mT5): READY                      │
│  ✅ P2 (SQLCoder): READY                 │
│  ✅ P3 (Vanna AI): READY                 │
└──────────────────────────────────────────┘
                 │
                 ↓
┌──────────────────────────────────────────┐
│  Run CELL 8: Start Server                │
│  ⏱️  Wait 10 seconds                      │
│                                          │
│  Expected output:                        │
│  🌐 PUBLIC URL:                          │
│  https://abnormally-...ngrok-free.app    │
│                                          │
│  📡 API Endpoints:                       │
│  /p1/generate (POST) ✅                  │
│  /p2/generate (POST) ✅                  │
│  /p3/generate (POST) ✅                  │
└──────────────────────────────────────────┘
                 │
                 ↓
┌──────────────────────────────────────────┐
│  🎉 API READY!                           │
│  Test from local frontend or curl       │
└──────────────────────────────────────────┘
```

---

## 🧪 **Quick Test:**

Once server is running, test in a new terminal:

```bash
# Replace with your actual ngrok URL
export API_URL="https://abnormally-direct-rhino.ngrok-free.app"

# Test health check
curl $API_URL/health

# Test P1 (mT5)
curl -X POST "$API_URL/p1/generate" \
     -H "Content-Type: application/json" \
     -d '{"query": "Tìm áo thun"}'

# Test P2 (SQLCoder)
curl -X POST "$API_URL/p2/generate" \
     -H "Content-Type: application/json" \
     -d '{"query": "Hiển thị giày"}'

# Test P3 (Vanna AI)
curl -X POST "$API_URL/p3/generate" \
     -H "Content-Type: application/json" \
     -d '{"query": "Top 10 sản phẩm bán chạy"}'
```

---

## 🎯 **Expected Timeline:**

```
00:00 - Start CELL 7
00:30 - P1 (mT5) loaded ✅
02:00 - P2 (SQLCoder) loaded ✅
02:30 - P3 (Vanna AI) loaded ✅
02:40 - All pipelines ready
02:50 - Start CELL 8
03:00 - API live! 🚀
```

**Total: ~3 minutes from start to API availability!**

---

## ⚡ **Frontend Integration:**

Once API is running, update your frontend (if not auto-detected):

```javascript
// frontend/.env.local or config
const COLAB_API_URL = "https://abnormally-direct-rhino.ngrok-free.app";

// Test search
fetch(`${COLAB_API_URL}/p1/generate`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'Tìm giày thể thao' })
})
.then(res => res.json())
.then(data => console.log(data.sql_query));
```

---

## 🔧 **Troubleshooting:**

### **Issue: P2 Not Loading**

```
[✗] Failed to initialize P2: CUDA out of memory
```

**Solution:**
```python
# In CELL 7, before running:
import torch
torch.cuda.empty_cache()

# Or restart runtime:
Runtime → Restart Runtime
```

---

### **Issue: P3 Not Loading**

```
[✗] OpenAI API key not found
```

**Solution:**
```python
# Option 1: Set in Colab Secrets
# Left sidebar → Secrets → Add "OPENAI_API_KEY"

# Option 2: Set MANUAL_API_KEY
# Top of notebook, line 31:
MANUAL_API_KEY = "sk-proj-..."
```

---

### **Issue: ngrok Tunnel Failed**

```
Error: ngrok authentication failed
```

**Solution:**
```python
# CELL 8 has auth token hardcoded
# If it fails, get new token from:
# https://dashboard.ngrok.com/get-started/your-authtoken

# Update line 4238:
ngrok.set_auth_token("YOUR_NEW_TOKEN")
```

---

## 📊 **What Gets Loaded:**

### **Models Downloaded (First Time):**

| Pipeline | Model | Size | Time |
|----------|-------|------|------|
| P1 | google/mt5-base | ~500 MB | ~1 min |
| P2 | defog/sqlcoder-7b-2 | ~14 GB | ~3 min |
| P3 | (Uses OpenAI API) | N/A | ~30 sec |

**Subsequent runs:** Models cached, ~30-60 sec total load time

---

## 💾 **GPU Memory Usage:**

```
P1 (mT5):           ~2.5 GB
P2 (SQLCoder 8-bit): ~7.0 GB
P3 (Vanna AI):       ~0.5 GB
──────────────────────────────
Total:              ~10.0 GB

✅ Fits on Colab T4 GPU (16 GB total)
```

---

## ✅ **Success Indicators:**

You'll know it's working when you see:

**CELL 7 Output:**
```
============================================================
Pipeline Initialization Summary:
============================================================
  P1 (mT5):       ✅ READY
  P2 (SQLCoder):  ✅ READY
  P3 (Vanna AI):  ✅ READY
============================================================
```

**CELL 8 Output:**
```
🌐 ngrok tunnel established!
📡 PUBLIC URL: https://abnormally-direct-rhino.ngrok-free.app

============================================================
🚀 API SERVER RUNNING!
============================================================
```

**Health Check:**
```bash
$ curl https://...ngrok-free.app/health
{
  "status": "healthy",
  "pipelines": {
    "P1": true,
    "P2": true,
    "P3": true
  }
}
```

---

## 🎉 **You're Done!**

Your Vietnamese NL2SQL API is now:
- ✅ Running on Colab
- ✅ Accessible via ngrok public URL
- ✅ All 3 pipelines exposed
- ✅ Ready for frontend calls
- ✅ No evaluation overhead!

**Next:** Test from your local frontend at `http://localhost:3000` or use curl commands above.

---

**Quick Reference:**
- **Notebook:** `colab_all_pipelines.py`
- **Run:** CELL 7 → CELL 8
- **Time:** ~3 minutes
- **Memory:** ~10 GB GPU
- **API:** Public via ngrok
- **Endpoints:** `/p1/generate`, `/p2/generate`, `/p3/generate`

🚀 **Happy Coding!**
