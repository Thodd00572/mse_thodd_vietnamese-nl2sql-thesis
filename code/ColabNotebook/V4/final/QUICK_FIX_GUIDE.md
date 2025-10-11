# Quick Fix Guide - Copy These Exact Changes to Colab

## 🎯 Three Changes to Make

---

## ✏️ CHANGE #1: Line 2043 (P2 Evaluation)

**Search for this line:**
```python
if __name__ == "__main__":
```

**Replace the 2 lines:**
```python
if __name__ == "__main__":
    metrics_p2, results_p2, pipeline_p2 = main()
```

**With these 2 lines:**
```python
# Changed from if __name__ == "__main__": to allow execution in Colab
metrics_p2, results_p2, pipeline_p2 = main()
```

---

## ✏️ CHANGE #2: Line 4241 (Preserve Pipelines)

**Search for these 7 lines:**
```python
# Initialize global pipeline variables (will be set when evaluation cells run)
pipeline_p1 = None
pipeline_p2 = None
pipeline_p3 = None
metrics_p1 = None
metrics_p2 = None
metrics_p3 = None
```

**Replace with these 37 lines:**
```python
# Initialize global pipeline variables (will be set when evaluation cells run)
# IMPORTANT: Only initialize if not already loaded from evaluation cells
try:
    if 'pipeline_p1' not in dir():
        pipeline_p1 = None
except NameError:
    pipeline_p1 = None

try:
    if 'pipeline_p2' not in dir():
        pipeline_p2 = None
except NameError:
    pipeline_p2 = None

try:
    if 'pipeline_p3' not in dir():
        pipeline_p3 = None
except NameError:
    pipeline_p3 = None

try:
    if 'metrics_p1' not in dir():
        metrics_p1 = None
except NameError:
    metrics_p1 = None

try:
    if 'metrics_p2' not in dir():
        metrics_p2 = None
except NameError:
    metrics_p2 = None

try:
    if 'metrics_p3' not in dir():
        metrics_p3 = None
except NameError:
    metrics_p3 = None
```

---

## ✏️ CHANGE #3: Line 4751 (Status Check)

**Search for these 4 lines:**
```python
print(f"\n{'='*60}")
print("STARTING FASTAPI SERVER")
print(f"{'='*60}")
print(f"Port: 8000")
```

**INSERT THIS CODE BEFORE THEM:**
```python
# CHECK PIPELINE STATUS BEFORE STARTING SERVER
print(f"\n{'='*60}")
print("PIPELINE STATUS CHECK")
print(f"{'='*60}")

p1_status = "✅ LOADED" if pipeline_p1 is not None else "❌ NOT LOADED"
p2_status = "✅ LOADED" if pipeline_p2 is not None else "❌ NOT LOADED"
p3_status = "✅ LOADED" if pipeline_p3 is not None else "❌ NOT LOADED"

print(f"P1 (mT5):       {p1_status}")
print(f"P2 (SQLCoder):  {p2_status}")
print(f"P3 (Vanna AI):  {p3_status}")

loaded_count = sum([pipeline_p1 is not None, pipeline_p2 is not None, pipeline_p3 is not None])
if loaded_count == 0:
    print("\n⚠️  WARNING: NO PIPELINES LOADED!")
    print("   You must run the evaluation cells (CELL 6) for each pipeline first:")
    print("   1. Find and run the P1 evaluation cell (around line 594)")
    print("   2. Find and run the P2 evaluation cell (around line 1409)")
    print("   3. Find and run the P3 evaluation cell (around line 3968)")
    print("\n   The API will start, but all /generate endpoints will fail!")
elif loaded_count < 3:
    print(f"\n⚠️  WARNING: Only {loaded_count}/3 pipelines loaded!")
    print("   Some endpoints will not work. Load missing pipelines first.")
else:
    print("\n✅ All pipelines loaded and ready!")

```

**Result:** The existing "STARTING FASTAPI SERVER" section stays, just add the status check before it.

---

## 📋 Complete Workflow

1. ✅ Make Change #1 (line 2043)
2. ✅ Make Change #2 (line 4241) 
3. ✅ Make Change #3 (line 4751)
4. ✅ Run CELL 6 for P1 (line 594)
5. ✅ Run CELL 6 for P2 (line 1409)
6. ✅ Run CELL 6 for P3 (line 3968)
7. ✅ Run CELL 7 (API setup)
8. ✅ Run CELL 8 (Start server)

You should see:
```
P1 (mT5):       ✅ LOADED
P2 (SQLCoder):  ✅ LOADED
P3 (Vanna AI):  ✅ LOADED

✅ All pipelines loaded and ready!
```

---

## 🔍 How to Find the Lines in Colab

### **Finding Line 2043:**
- Press `Ctrl+F` (or `Cmd+F` on Mac)
- Search: `if __name__ == "__main__":`
- Should find the line in the P2 section

### **Finding Line 4241:**
- Search: `# Initialize global pipeline variables`
- Look in CELL 7 section
- Should be right after `ngrok.set_auth_token`

### **Finding Line 4751:**
- Search: `STARTING FASTAPI SERVER`
- Look in CELL 8 section
- Add the status check RIGHT BEFORE this print statement

---

## ✅ Verification Command

After making all changes and running the cells, run this test:

```python
# Run this in a new cell AFTER CELL 8
print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)
print(f"P1 loaded: {pipeline_p1 is not None}")
print(f"P2 loaded: {pipeline_p2 is not None}")
print(f"P3 loaded: {pipeline_p3 is not None}")
print(f"\nAll ready: {all([pipeline_p1 is not None, pipeline_p2 is not None, pipeline_p3 is not None])}")
```

Expected output:
```
P1 loaded: True
P2 loaded: True
P3 loaded: True

All ready: True
```

---

**Done!** Your Colab notebook is now fixed and ready to use. 🎉
