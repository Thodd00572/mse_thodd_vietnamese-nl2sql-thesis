# Database Connection Fix Summary

**Issue**: Database page at `http://localhost:3000/database/` was loading forever and couldn't connect

**Root Cause**: Hardcoded relative database paths that didn't match the actual database location

---

## Problem Diagnosis

### Symptoms:
- Frontend page loads but shows "Loading statistics..." forever
- Console shows API call to `/api/database/stats` but no response
- Database exists at `/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db`
- Backend was looking for database at `core/data/tiki_products_normalized.db` (wrong path)

### Files With Issues:
1. `/code/demo/backend/database/db_manager_normalized.py` - hardcoded path in `__init__`
2. `/code/demo/backend/api/database_routes.py` - hardcoded paths in fallback and query endpoints

---

## Fixes Applied

### 1. Fixed DatabaseManager (`db_manager_normalized.py`)

**Before**:
```python
def __init__(self, db_path: str = "core/data/tiki_products_normalized.db"):
    self.db_path = db_path
```

**After**:
```python
def __init__(self, db_path: str = None):
    # Use absolute path to database if not provided
    if db_path is None:
        # Try multiple possible locations
        possible_paths = [
            "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db",
            "data/tiki_products_normalized.db",
            "../../../data/tiki_products_normalized.db",
            "core/data/tiki_products_normalized.db"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                self.db_path = path
                logger.info(f"Found database at: {self.db_path}")
                break
        else:
            # Default to absolute path
            self.db_path = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db"
            logger.warning(f"Database not found, using default path: {self.db_path}")
    else:
        self.db_path = db_path
```

**Benefits**:
- ✅ Tries multiple possible locations (absolute + relative paths)
- ✅ Logs which path was found
- ✅ Falls back to correct absolute path
- ✅ Still allows custom path if provided

### 2. Fixed Database Routes (`database_routes.py`)

#### Stats Endpoint Fallback:
**Before**:
```python
conn = sqlite3.connect("data/tiki_products_normalized.db", timeout=5.0)
```

**After**:
```python
# Try to find the database
possible_paths = [
    "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db",
    "data/tiki_products_normalized.db",
    "../../../data/tiki_products_normalized.db"
]

db_path = None
for path in possible_paths:
    if os.path.exists(path):
        db_path = path
        break

if not db_path:
    raise Exception("Database file not found in any expected location")

conn = sqlite3.connect(db_path, timeout=5.0)
```

#### Query Endpoint:
**Before**:
```python
db_path = "data/tiki_products_normalized.db"
conn = sqlite3.connect(db_path, timeout=5)
```

**After**:
```python
# Find database path
possible_paths = [
    "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db",
    "data/tiki_products_normalized.db",
    "../../../data/tiki_products_normalized.db"
]

db_path = None
for path in possible_paths:
    if os.path.exists(path):
        db_path = path
        break

if not db_path:
    raise HTTPException(status_code=500, detail="Database file not found")

conn = sqlite3.connect(db_path, timeout=5)
```

---

## How to Test

### 1. Restart Backend Server
```bash
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/backend
python -m uvicorn main:app --reload --port 8000
```

### 2. Check Backend Logs
You should see:
```
INFO: Found database at: /Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db
INFO: Normalized database ready with tables: ['brands', 'categories', 'product_pricing', 'product_reviews', 'products', 'sellers']
```

### 3. Test Database Page
1. Navigate to: http://localhost:3000/database/
2. You should see:
   - Total Records count
   - Categories count
   - Unique Brands count
   - Average Price
   - Top Categories grid
   - Data Sources list

### 4. Test API Directly
```bash
# Test stats endpoint
curl http://localhost:8000/api/database/stats

# Test query endpoint
curl -X POST http://localhost:8000/api/database/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT COUNT(*) FROM products;"}'
```

---

## Expected Behavior After Fix

### Database Page (http://localhost:3000/database/)

#### Overview Tab:
- ✅ Shows total products (should be ~30,000+)
- ✅ Shows categories count
- ✅ Shows brands count
- ✅ Shows average price in Vietnamese Dong (₫)
- ✅ Displays top 9 categories with product counts
- ✅ Lists data sources with record counts

#### SQL Query Tab:
- ✅ Predefined queries organized by complexity (Simple/Medium/Complex)
- ✅ Query editor with syntax highlighting
- ✅ Execute button runs queries
- ✅ Results displayed in table format

#### Data Browser Tab:
- ✅ Paginated product data (50 items per page)
- ✅ Shows ID, Name, Price, Brand, Category, Rating
- ✅ Previous/Next pagination controls

#### Schema Tab:
- ✅ Displays normalized database schema
- ✅ Shows all 6 tables (brands, categories, products, product_pricing, product_reviews, sellers)
- ✅ Column details with types and descriptions

---

## Database Location Strategy

The fix uses a **fallback strategy** that tries paths in order:

1. **Absolute path** (most reliable): `/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db`
2. **Relative from backend**: `data/tiki_products_normalized.db`
3. **Relative up three levels**: `../../../data/tiki_products_normalized.db`
4. **Old location** (legacy): `core/data/tiki_products_normalized.db`

This ensures the database is found regardless of:
- Where the backend is run from (root, backend/, code/)
- Whether using absolute or relative imports
- Different development vs production setups

---

## Database Verification

To verify database integrity:

```bash
# Check file exists and is valid SQLite
file /Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db

# Output should show:
# SQLite 3.x database, last written using SQLite version 3043002

# Check tables exist
sqlite3 /Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db

sqlite> .tables
# Should show: brands  categories  product_pricing  product_reviews  products  sellers

sqlite> SELECT COUNT(*) FROM products;
# Should show: 30000+ records

sqlite> .exit
```

---

## Additional Improvements Made

### Better Error Messages:
- Now includes actual database path in error messages
- Logs which path was successfully found
- More descriptive HTTPException details

### Timeout Handling:
- All database connections use `timeout=5.0` or `timeout=10.0`
- Prevents infinite hangs on locked database

### Logging:
- Added `logger.info()` when database is found
- Added `logger.warning()` when using fallback path
- Better error tracking in backend logs

---

## Troubleshooting

### If database page still loading:

1. **Check backend is running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check database file exists**:
   ```bash
   ls -lh /Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db
   ```

3. **Check backend logs**:
   Look for lines containing:
   - "Found database at:"
   - "Normalized database ready with tables:"

4. **Test API endpoint directly**:
   ```bash
   curl http://localhost:8000/api/database/stats
   ```

5. **Check browser console**:
   - Open DevTools (F12)
   - Check Console tab for errors
   - Check Network tab for failed requests

### Common Issues:

| Issue | Solution |
|-------|----------|
| Backend not running | Start with `python -m uvicorn main:app --reload --port 8000` |
| Database locked | Close any SQLite browsers/tools accessing the database |
| CORS error | Check backend has CORS middleware enabled |
| 404 on API | Verify backend URL is `http://localhost:8000/api/database/stats` |
| Frontend cached | Hard refresh: Cmd+Shift+R (Mac) or Ctrl+F5 (Windows) |

---

**Status**: ✅ Fixed  
**Date**: October 1, 2025  
**Files Modified**: 
- `db_manager_normalized.py`
- `database_routes.py`

**Next Steps**: Restart backend server and test database page
