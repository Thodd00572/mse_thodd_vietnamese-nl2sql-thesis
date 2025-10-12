# Enrichment Diagnostic Steps

**Date**: October 12, 2025, 11:55 PM  
**Issue**: Category, rating, price not loading in search results  
**Status**: 🔍 **DIAGNOSING**

---

## Problem

Search results show:
- ❌ "Price not available"
- ❌ "Uncategorized"
- ❌ "No ratings"

This means backend enrichment is **not working**.

---

## Step 1: Restart Backend Server

**CRITICAL**: You must restart the backend to load the new enrichment code!

```bash
# Stop current backend (if running)
# Press Ctrl+C in the terminal running the server

# Start backend
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/backend
./scripts/start_api.sh

# OR manually:
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/backend
uvicorn core.main:app --reload --host 0.0.0.0 --port 8000
```

**Watch for this log message**:
```
✅ Using database: /Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db
```

---

## Step 2: Test Enrichment Endpoint

Open your browser or use curl to test the diagnostic endpoint:

### Browser:
```
http://localhost:8000/api/debug/enrichment-test
```

### Terminal:
```bash
curl http://localhost:8000/api/debug/enrichment-test
```

---

## Step 3: Interpret Results

### ✅ SUCCESS Response:
```json
{
  "status": "success",
  "database_path": "/Users/thoduong/.../data/tiki_products_normalized.db",
  "enrichment_fields": {
    "brand": "Baellerry",           // ← Should have value
    "category": "Thời Trang",       // ← Should have value
    "price": 299000,                // ← Should be number
    "rating": 4.5                   // ← Should be number (or null)
  },
  "raw_fields": ["product_id", "name", "category_id", ...],
  "enriched_fields": ["product_id", "name", "brand", "category", "price", ...]
}
```

**If you see this**: Enrichment is working! Problem is elsewhere.

---

### ❌ FAILURE Responses:

#### Error 1: Wrong Database Path
```json
{
  "database_path": "/Users/.../data/tiki_products_normalized.db",
  "error": "No products found in database"
}
```

**Fix**: Database file doesn't exist or path is wrong
```bash
# Verify database exists
ls -lh /Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db

# If not found, check where it actually is
find /Users/thoduong/CascadeProjects/MSE_Thesis_2025 -name "*.db" -o -name "*.sqlite"
```

---

#### Error 2: Enrichment Returns "MISSING"
```json
{
  "enrichment_fields": {
    "brand": "MISSING",
    "category": "MISSING",
    "price": "MISSING",
    "rating": "MISSING"
  }
}
```

**Fix**: Enrichment logic failed silently
- Check backend logs for errors
- Verify database has related tables (brands, categories, product_pricing, product_reviews)

---

#### Error 3: Import Error
```json
{
  "error": "name 'enrich_products_with_details' is not defined",
  "error_type": "NameError"
}
```

**Fix**: Backend not restarted with new code
- Stop backend (Ctrl+C)
- Restart backend
- Try again

---

## Step 4: Check Backend Logs

After running a search query, check backend logs for these messages:

### Expected Logs (Working):
```
INFO: 🔍 Executing SQL: SELECT * FROM products WHERE name LIKE '%ví%' LIMIT 10...
INFO: 📊 Got 10 raw results
INFO: 🔎 Checking if enrichment needed. First row keys: ['product_id', 'name', 'category_id', ...]
INFO: ✅ product_id found! Starting enrichment for 10 products...
INFO: Starting enrichment for 10 products
INFO:   🏷️ Looking up category_id=17
INFO:   ✅ Category enriched: Thời Trang
INFO: ✅ Successfully enriched 10 products
```

### Error Logs (Not Working):
```
INFO: ⏭️ Skipping enrichment - no product_id in results
```
OR
```
ERROR: ❌ Enrichment failed: [error message]
```

---

## Step 5: Manual Database Test

Test the database directly:

```bash
# Connect to database
sqlite3 /Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db

# Test product query
SELECT product_id, name, category_id FROM products WHERE name LIKE '%ví%' LIMIT 3;

# Test category lookup
SELECT * FROM categories WHERE category_id = 17;

# Test brand lookup
SELECT * FROM brands LIMIT 5;

# Test price lookup
SELECT * FROM product_pricing LIMIT 5;

# Test rating lookup
SELECT * FROM product_reviews LIMIT 5;

# Exit
.exit
```

**Expected Results**:
- Products should have valid category_id (e.g., 17)
- Categories table should have category_id=17 with name
- All tables should have data

---

## Step 6: Check Frontend Logs

Open browser DevTools (F12) → Console tab

Look for:
- Network errors (failed API calls)
- JavaScript errors
- Response data structure

---

## Common Issues & Fixes

### Issue 1: Backend Not Restarted
**Symptom**: Enrichment test endpoint doesn't exist (404)  
**Fix**: Restart backend server

### Issue 2: Wrong Database Path
**Symptom**: Enrichment test shows empty database  
**Fix**: Update `database/db_manager_normalized.py` with correct path

### Issue 3: Import Path Wrong
**Symptom**: NameError or ImportError  
**Fix**: Check imports in `api/routes.py` - should import from `shared_models`

### Issue 4: Database Missing Tables
**Symptom**: SQL errors in logs  
**Fix**: Verify database schema - check if brands, categories, product_pricing, product_reviews tables exist

### Issue 5: Python Path Issues
**Symptom**: Cannot import modules  
**Fix**: Check sys.path setup in routes.py

---

## Quick Checklist

- [ ] Backend restarted after code changes
- [ ] Diagnostic endpoint returns success
- [ ] Database path is correct
- [ ] Database tables exist (products, brands, categories, product_pricing, product_reviews)
- [ ] Backend logs show enrichment messages
- [ ] Frontend receives enriched data

---

## Debug Commands Reference

```bash
# 1. Check if backend is running
lsof -i :8000

# 2. Restart backend
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/backend
./scripts/start_api.sh

# 3. Test enrichment endpoint
curl http://localhost:8000/api/debug/enrichment-test | jq

# 4. Check database file
ls -lh /Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db

# 5. Test database query
sqlite3 /Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db \
  "SELECT COUNT(*) FROM products"

# 6. View backend logs (if running in background)
tail -f /path/to/backend/logs

# 7. Test actual search API
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Tìm ví", "pipeline": "pipeline1"}' | jq '.pipeline1_result.results[0]'
```

---

## Expected Workflow

1. ✅ **Restart backend** → See database path log
2. ✅ **Test enrichment endpoint** → Get success response with enriched data
3. ✅ **Search for products** → See enriched data in results
4. ✅ **Check frontend** → Price, category, brand, rating all display

---

## If Still Not Working

### Last Resort Debugging:

1. **Add print statements** in `core/shared_models.py` enrichment function
2. **Check Python environment** - make sure correct Python version
3. **Verify file permissions** - database file must be readable
4. **Check for multiple backend instances** - kill all and restart one
5. **Clear browser cache** - hard refresh (Ctrl+Shift+R)

---

## Success Criteria

When everything works, you should see:

### In Diagnostic Endpoint:
```json
{
  "status": "success",
  "enrichment_fields": {
    "brand": "Baellerry",
    "category": "Thời Trang",
    "price": 299000,
    "rating": 4.5
  }
}
```

### In Search Results:
- ✅ Product name displayed
- ✅ Brand name with 🏷️ icon (blue)
- ✅ Price in large green text (₫299,000)
- ✅ Category in purple badge (📂 Thời Trang)
- ✅ Rating with ⭐ (4.5 stars)

---

**Next Step**: Restart backend and run the diagnostic endpoint!

---

**Created**: October 12, 2025, 11:55 PM  
**Diagnostic Endpoint**: `http://localhost:8000/api/debug/enrichment-test`  
**Purpose**: Identify why enrichment is not working
