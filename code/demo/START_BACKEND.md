# Start Backend Server - Quick Guide

## ❌ Current Problem:
Backend server on port 8000 is not running → "Network Error" when clicking Search

## ✅ Solution: Start Backend Manually

### **Step 1: Open New Terminal**
Open a **separate terminal window** (keep it running)

### **Step 2: Navigate to Backend Directory**
```bash
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/backend/core
```

### **Step 3: Start the Server**
```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### **Step 4: Wait for Success Message**
You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### **Step 5: Verify Server is Running**
Open browser: http://localhost:8000/docs
- Should see FastAPI Swagger UI

### **Step 6: Test Search Again**
- Go back to frontend: http://localhost:3000
- Click any sample query (e.g., "Tìm áo thun")
- Should work now!

---

## 🔧 If You Get Errors:

### **Error: "Address already in use"**
Kill the old process:
```bash
lsof -i :8000
# Note the PID number
kill -9 <PID>
# Then retry Step 3
```

### **Error: "ModuleNotFoundError"**
Install dependencies:
```bash
cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/backend
pip3 install -r requirements.txt
# Then retry Step 3
```

### **Error: "No module named 'uvicorn'"**
Install uvicorn:
```bash
pip3 install uvicorn fastapi
# Then retry Step 3
```

---

## ✅ Quick Verification Commands:

```bash
# Check if port 8000 is in use
lsof -i :8000

# Test backend API
curl http://localhost:8000/docs

# Check health endpoint
curl http://localhost:8000/api/health
```

---

## 📝 Keep Backend Running:
- **DO NOT close the terminal** where backend is running
- You should see log messages when you search
- Press CTRL+C to stop the server

---

## 🎯 Summary:
1. Open separate terminal
2. `cd /Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/demo/backend/core`
3. `python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
4. Wait for "Application startup complete"
5. Try search again!
