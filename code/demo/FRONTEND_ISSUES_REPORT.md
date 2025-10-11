# Frontend Home Page - Issues Report

## 📊 Dataset Analysis

**File:** `/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/ColabNotebook/data/eval_data.jsonl`

### **Total Queries:** 300
- **Simple queries:** 100 (index 0-99)
- **Medium queries:** 100 (index 100-199)  
- **Complex queries:** 100 (index 200-299)

---

## ✅ **What's Working**

### **1. Colab Endpoint Configuration** (config.js)
```javascript
base_url: 'https://abnormally-direct-rhino.ngrok-free.app'
pipeline1_url: '/p1'  ✅
pipeline2_url: '/p2'  ✅
pipeline3_url: '/p3'  ✅
```
**Status:** ✅ Correctly configured

---

### **2. Search Function** (index.js lines 34-137)
```javascript
const handleSearch = async (e) => {
  const response = await api.post('/api/search', {
    query: query.trim(),
    pipeline: selectedPipeline
  })
  setResults(response.data)
}
```
**Status:** ✅ Working properly

---

## ❌ **Critical Issue: Sample Queries Mismatch**

### **Problem:**
Sample queries in `index.js` (lines 840-918) are **HARDCODED** and **DO NOT match** `eval_data.jsonl`

### **Evidence:**

#### **Hardcoded in Frontend (index.js):**
```javascript
// Simple Queries (9 samples)
"Tìm tất cả balo nữ",
"Hiển thị các giày thể thao nam có giá dưới 500k",
"Tìm túi xách của thương hiệu Samsonite",
"Cho tôi xem tất cả dép tổ ong",
...

// Medium Queries (8 samples)
"Tìm giày thể thao thương hiệu Nike có giá dưới 1 triệu",
"Hiển thị balo nam hoặc balo nữ có rating trên 4.5 sao",
...

// Complex Queries (7 samples)
"Tìm giày có giá cao nhất trong danh mục Giày thể thao nam",
"Hiển thị top 10 balo bán chạy nhất có rating trên 4.5 sao",
...
```

#### **Actual eval_data.jsonl:**
```jsonl
// Simple (index 0-99)
{"index": 0, "vietnamese": "Tìm áo thun", ...}
{"index": 1, "vietnamese": "Hiển thị giày", ...}
{"index": 2, "vietnamese": "Xem túi xách", ...}
{"index": 6, "vietnamese": "Hiển thị dép", ...}
...

// Medium (index 100-199)
{"index": 103, "vietnamese": "Thương hiệu Nike", ...}
{"index": 104, "vietnamese": "Sản phẩm giá dưới 500k", ...}
{"index": 117, "vietnamese": "Sản phẩm thương hiệu Nike", ...}
...

// Complex (index 200-299)
{"index": 200, "vietnamese": "Top 10 sản phẩm đánh giá cao nhất có giá dưới 1 triệu", ...}
{"index": 201, "vietnamese": "Phân tích thị phần thương hiệu", ...}
{"index": 202, "vietnamese": "Top 5 sản phẩm bán chạy nhất trong danh mục Phụ kiện thời trang", ...}
...
```

### **Mismatch Details:**

| Frontend Hardcoded | eval_data.jsonl Actual | Match? |
|-------------------|------------------------|--------|
| "Tìm tất cả balo nữ" | "Tìm áo thun" (index 0) | ❌ NO |
| "Hiển thị các giày thể thao nam có giá dưới 500k" | "Hiển thị giày" (index 1) | ❌ NO |
| "Tìm túi xách của thương hiệu Samsonite" | "Xem túi xách" (index 2) | ❌ NO |
| "Cho tôi xem tất cả dép tổ ong" | "Hiển thị dép" (index 6) | ❌ NO |

**Result:** 0 out of 24 sample queries match the evaluation dataset exactly!

---

## 🔴 **Impact of This Issue**

1. **Inconsistent Testing:** Users test with queries NOT in the evaluation set
2. **Misleading Performance:** Results don't reflect actual pipeline performance on eval_data
3. **No Reproducibility:** Cannot verify results against known evaluation queries
4. **Research Integrity:** Sample queries should match the dataset used for metrics

---

## ✅ **Recommended Solution**

### **Option 1: Dynamic Sample Loading from Backend (Recommended)**

#### **Step 1: Add Backend API Endpoint**
Create `/api/samples` endpoint that:
- Reads `eval_data.jsonl`
- Returns random samples per complexity
- Format:
```json
{
  "simple": [
    {"vietnamese": "Tìm áo thun", "index": 0},
    {"vietnamese": "Hiển thị giày", "index": 1},
    ...
  ],
  "medium": [...],
  "complex": [...]
}
```

#### **Step 2: Update Frontend to Fetch Samples**
```javascript
const [sampleQueries, setSampleQueries] = useState({
  simple: [],
  medium: [],
  complex: []
})

useEffect(() => {
  fetchSampleQueries()
}, [])

const fetchSampleQueries = async () => {
  const response = await api.get('/api/samples')
  setSampleQueries(response.data)
}
```

#### **Step 3: Render Dynamic Samples**
```javascript
{sampleQueries.simple.map((sample, idx) => (
  <button
    key={`simple-${idx}`}
    onClick={() => setQuery(sample.vietnamese)}
  >
    "{sample.vietnamese}"
  </button>
))}
```

---

### **Option 2: Update Hardcoded Samples to Match eval_data.jsonl**

Replace lines 840-918 in `index.js` with actual queries from `eval_data.jsonl`:

#### **Simple (from index 0-20):**
```javascript
[
  "Tìm áo thun",
  "Hiển thị giày",
  "Xem túi xách",
  "Liệt kê balo",
  "Tìm kiếm vali",
  "Tìm ví",
  "Hiển thị dép",
  "Xem nón",
  "Liệt kê thắt lưng",
  "Tìm kiếm kính"
]
```

#### **Medium (from index 100-110):**
```javascript
[
  "Sản phẩm theo thương hiệu",
  "Giá trung bình theo danh mục",
  "Sản phẩm có đánh giá cao",
  "Thương hiệu Nike",
  "Sản phẩm giá dưới 500k",
  "Sản phẩm giá dưới 100k",
  "Sản phẩm giá trên 100k",
  "Sản phẩm giá dưới 200k"
]
```

#### **Complex (from index 200-210):**
```javascript
[
  "Top 10 sản phẩm đánh giá cao nhất có giá dưới 1 triệu",
  "Phân tích thị phần thương hiệu",
  "Top 5 sản phẩm bán chạy nhất trong danh mục Phụ kiện thời trang",
  "Top 6 sản phẩm bán chạy nhất trong danh mục Giày dép nam",
  "Top 7 sản phẩm bán chạy nhất trong danh mục Túi nam",
  "Top 8 sản phẩm bán chạy nhất trong danh mục Balo & Vali",
  "Top 9 sản phẩm bán chạy nhất trong danh mục Phụ kiện thời trang"
]
```

---

## 📋 **Action Items**

- [ ] **Priority 1:** Implement dynamic sample loading (Option 1) OR update hardcoded samples (Option 2)
- [ ] **Priority 2:** Add "Shuffle Samples" button to load different examples
- [ ] **Priority 3:** Add sample index display (e.g., "Sample #5 from eval_data.jsonl")
- [ ] **Priority 4:** Add link to view full eval_data.jsonl

---

## 🎯 **Summary**

| Component | Status | Issue |
|-----------|--------|-------|
| **Colab Endpoints** | ✅ OK | None |
| **Search Function** | ✅ OK | None |
| **Sample Queries** | ❌ FAIL | Not from eval_data.jsonl |

**Recommendation:** Use **Option 1 (Dynamic Loading)** for better maintainability and alignment with the evaluation dataset.

---

**Report Generated:** 2025-10-10  
**eval_data.jsonl Location:** `/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/ColabNotebook/data/eval_data.jsonl`  
**Total Queries:** 300 (100 simple, 100 medium, 100 complex)
