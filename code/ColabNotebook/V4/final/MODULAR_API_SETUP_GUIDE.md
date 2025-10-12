# Modular API Setup Guide - Vietnamese NL2SQL Pipelines

## 📋 Overview

This guide explains how to use the **modular import approach** to run all 3 pipelines (P1, P2, P3) with their complete logic via API.

### ✅ Benefits of Modular Approach

1. **Single Source of Truth** - Each pipeline's logic lives in ONE file
2. **Easy Updates** - Edit one file, changes apply everywhere
3. **No Code Duplication** - Cleaner and more maintainable
4. **Always Latest Logic** - Imports ensure you use the most recent code
5. **Post-processing Included** - All helper methods are imported automatically

---

## 📁 File Structure

```
/content/  (Google Colab)
├── colab_p1_mt5_zero.py          # P1 complete logic
├── colab_p2_sqlcoder_zero.py     # P2 complete logic
├── colab_p3_vanna_zero.py        # P3 complete logic
├── colab_api_server.py           # NEW: API server (imports above)
└── vietnamese_tiki_products.db   # Database (for P3)
```

---

## 🚀 Setup Instructions

### **Step 1: Upload Files to Google Colab**

Upload these 5 files to your Colab session:

1. ✅ `colab_p1_mt5_zero.py` - P1 mT5 pipeline
2. ✅ `colab_p2_sqlcoder_zero.py` - P2 SQLCoder pipeline
3. ✅ `colab_p3_vanna_zero.py` - P3 Vanna AI pipeline
4. ✅ `colab_api_server.py` - **NEW** API server
5. ✅ `vietnamese_tiki_products.db` - Database

### **Step 2: Add OpenAI API Key to Colab Secrets**

For P3 (Vanna AI) to work:

1. Click the 🔑 **Secrets** icon in the left sidebar
2. Add a new secret:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key

### **Step 3: Run the API Server**

Open `colab_api_server.py` and run cells in order:

```python
# CELL 1: Install packages (run once)
install_packages()

# CELL 2: Import pipeline classes
# This imports PromptingPipeline, SQLCoderZeroShotPipeline, VannaPipeline

# CELL 3: Initialize pipelines
# Loads models and sets up P1, P2, P3

# CELL 4: FastAPI setup
# Configures API endpoints

# CELL 5: Start server
# Starts ngrok tunnel and FastAPI server
```

### **Step 4: Get Your Public URL**

After running Cell 5, you'll see:

```
✓ Public URL: https://your-unique-id.ngrok-free.app
  Use this URL in your frontend configuration
```

Copy this URL and update your local backend:

```bash
# Update the Colab URL in your local backend
# File: code/demo/backend/api/colab_proxy_routes.py
COLAB_BASE_URL = "https://your-unique-id.ngrok-free.app"
```

---

## 🔧 How It Works

### **Import Mechanism**

The `colab_api_server.py` file imports the pipeline classes:

```python
# Import P1 Pipeline
from colab_p1_mt5_zero import PromptingPipeline as P1Pipeline

# Import P2 Pipeline
from colab_p2_sqlcoder_zero import SQLCoderZeroShotPipeline as P2Pipeline

# Import P3 Pipeline
from colab_p3_vanna_zero import VannaPipeline as P3Pipeline
```

### **What Gets Imported**

When you import a class, **ALL its methods** come with it:

#### **P1 (mT5) - PromptingPipeline**
- ✅ `generate_sql()` - Main generation method
- ✅ `create_prompt()` - Prompt engineering
- ✅ `post_process_sql()` - SQL cleaning
- ✅ `extract_sql()` - SQL extraction
- ✅ `normalize_sql_format()` - Formatting
- ✅ `detect_count_intent()` - Intent detection
- ✅ `extract_search_term()` - Term extraction

#### **P2 (SQLCoder) - SQLCoderZeroShotPipeline**
- ✅ `generate_sql()` - Main generation method
- ✅ `create_prompt()` - Schema-aware prompting
- ✅ `post_process_sql()` - SQL validation
- ✅ `extract_sql()` - SQL extraction
- ✅ `detect_count_intent()` - Intent detection

#### **P3 (Vanna AI) - VannaPipeline**
- ✅ `generate_sql()` - RAG-based generation
- ✅ `setup_database_schema()` - Schema setup
- ✅ `generate_synthetic_training_data()` - Data augmentation
- ✅ `extract_query_patterns()` - Pattern extraction
- ✅ `configure_vanna_for_vietnamese()` - Vietnamese config
- ✅ `load_training_data()` - Training data loading
- ✅ All helper methods

---

## 🔄 Updating Pipeline Logic

### **To Update P1 Logic:**

1. Edit `colab_p1_mt5_zero.py`
2. Upload the updated file to Colab
3. Restart the kernel
4. Run `colab_api_server.py` again

The API will now use the updated P1 logic!

### **To Update P2 Logic:**

1. Edit `colab_p2_sqlcoder_zero.py`
2. Upload the updated file to Colab
3. Restart the kernel
4. Run `colab_api_server.py` again

### **To Update P3 Logic:**

1. Edit `colab_p3_vanna_zero.py`
2. Upload the updated file to Colab
3. Restart the kernel
4. Run `colab_api_server.py` again

---

## 🧪 Testing the API

### **Test P1 Endpoint:**

```bash
curl -X POST https://your-url.ngrok-free.app/p1/generate \
  -H "Content-Type: application/json" \
  -d '{"query":"Tìm áo thun"}'
```

**Expected Response:**
```json
{
  "pipeline": "P1_mT5_Zero_Shot",
  "vietnamese_query": "Tìm áo thun",
  "sql_query": "SELECT * FROM products WHERE name LIKE '%áo thun%' LIMIT 10;",
  "execution_time": 0.305,
  "valid": true,
  "success": true,
  "metrics": {
    "latency_ms": 305.2,
    "model": "google/mt5-base",
    "method": "zero_shot_prompting"
  }
}
```

### **Test P2 Endpoint:**

```bash
curl -X POST https://your-url.ngrok-free.app/p2/generate \
  -H "Content-Type: application/json" \
  -d '{"query":"Tìm giày"}'
```

### **Test P3 Endpoint:**

```bash
curl -X POST https://your-url.ngrok-free.app/p3/generate \
  -H "Content-Type: application/json" \
  -d '{"query":"Tìm túi xách"}'
```

### **Check Health:**

```bash
curl https://your-url.ngrok-free.app/health
```

**Expected Response:**
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

## ⚠️ Troubleshooting

### **Import Error: "No module named 'colab_p1_mt5_zero'"**

**Solution:** Make sure the file is uploaded to Colab's `/content/` directory

```python
# Check if files exist
import os
print(os.listdir('/content/'))
```

### **P3 Not Initializing**

**Possible causes:**
1. ❌ No OpenAI API key in Colab secrets
2. ❌ Database file not uploaded
3. ❌ Database path incorrect

**Solution:**
```python
# Check database exists
import os
db_path = "/content/vietnamese_tiki_products.db"
print(f"Database exists: {os.path.exists(db_path)}")

# Check API key
from google.colab import userdata
try:
    key = userdata.get('OPENAI_API_KEY')
    print("API key found!")
except:
    print("API key NOT found - add to Colab secrets")
```

### **GPU Out of Memory**

If you get OOM errors when loading all 3 pipelines:

**Option 1:** Run pipelines separately
- Comment out P1 and P2 initialization
- Only run P3

**Option 2:** Use smaller models
- P1: Use `google/mt5-small` instead of `google/mt5-base`
- P2: Use 4-bit quantization

---

## 📊 Comparison: Old vs New Approach

### **Old Approach (colab_all_pipelines.py):**

❌ **Problems:**
- 5000+ lines of duplicated code
- Hard to maintain
- Updates require editing multiple sections
- Post-processing might be missing
- Sync issues between files

### **New Approach (colab_api_server.py):**

✅ **Benefits:**
- ~300 lines of clean code
- Single source of truth
- Easy to update
- All methods included automatically
- No sync issues

---

## 🎯 Summary

### **What You Should Do:**

1. ✅ Use `colab_api_server.py` for API deployment
2. ✅ Keep pipeline logic in individual files (P1, P2, P3)
3. ✅ Update logic in individual files only
4. ✅ Restart kernel after updates

### **What You Should NOT Do:**

1. ❌ Don't use `colab_all_pipelines.py` for API (too large, duplicated)
2. ❌ Don't copy-paste code between files
3. ❌ Don't edit pipeline logic in multiple places

---

## 📝 Quick Start Checklist

- [ ] Upload `colab_p1_mt5_zero.py` to Colab
- [ ] Upload `colab_p2_sqlcoder_zero.py` to Colab
- [ ] Upload `colab_p3_vanna_zero.py` to Colab
- [ ] Upload `colab_api_server.py` to Colab
- [ ] Upload `vietnamese_tiki_products.db` to Colab
- [ ] Add `OPENAI_API_KEY` to Colab secrets
- [ ] Run Cell 1 in `colab_api_server.py` (install packages)
- [ ] Run Cell 2 (import pipelines)
- [ ] Run Cell 3 (initialize pipelines)
- [ ] Run Cell 4 (setup FastAPI)
- [ ] Run Cell 5 (start server)
- [ ] Copy ngrok URL
- [ ] Update local backend with ngrok URL
- [ ] Test endpoints with curl or frontend

---

## 🎓 For Your Thesis

When documenting your system architecture, you can explain:

> "The system uses a modular architecture where each pipeline's logic is maintained in a separate file. The API server imports these pipeline classes dynamically, ensuring that all methods including post-processing, validation, and error handling are automatically included. This approach provides a single source of truth for each pipeline's implementation and simplifies maintenance and updates."

---

**Need Help?**

If you encounter any issues, check:
1. Colab runtime logs for error messages
2. File upload status in Colab
3. GPU memory usage
4. API key configuration

Good luck with your thesis! 🎓✨
