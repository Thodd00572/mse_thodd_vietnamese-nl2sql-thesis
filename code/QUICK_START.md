# Quick Start Guide - Vietnamese NL2SQL Thesis

## For Busy Reviewers / Quick Setup

---

## What You Need to Know

**Best Pipeline**: P3 Vanna AI  
**Performance**: 76.3% Execution Accuracy (highest achieved)  
**Location**: `ColabNotebook/V4/final/colab_p3_vanna_zero.py`  

---

## 3-Minute Setup

### 1. Find the Production Code
```bash
cd ColabNotebook/V4/final/
```

You'll see **3 files**:
- `colab_p1_mt5_zero.py` - mT5 baseline (48% EM, 59% EX)
- `colab_p2_sqlcoder_zero.py` - SQLCoder baseline (19% EM, 21% EX)  
- `colab_p3_vanna_zero.py` - **Vanna AI BEST (43% EM, 76.3% EX)**

### 2. Upload to Google Colab
- Go to [Google Colab](https://colab.research.google.com/)
- Upload `colab_p3_vanna_zero.py`

### 3. Run It
```python
# In Colab, execute these cells in order:

# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Add OpenAI API Key (for P3 only)
api_key = "your-openai-api-key-here"

# Cell 3-6: Run all remaining cells
# Results auto-save to: /content/drive/MyDrive/vn2sql/logs/
```

### 4. View Results
Results include:
- Exact Match (EM): 43%
- Execution Accuracy (EX): 76.3%
- Detailed complexity breakdown
- Sample query results

---

## What Each Pipeline Does

| Pipeline | Model | Strategy | EM | EX | Speed |
|----------|-------|----------|----|----|-------|
| **P1** | Google mT5 | Zero-shot prompting | 48% | 59% | 0.30s |
| **P2** | SQLCoder-7B | Code-specialized LLM | 19% | 21% | 1.33s |
| **P3** | Vanna AI + GPT-4o | RAG retrieval | **43%** | **76.3%** | 1.78s |

---

## Where Everything Is

```
code/
├── ColabNotebook/V4/final/    ← Production code (START HERE)
├── ColabNotebook/data/        ← Evaluation data (300 queries)
├── ColabNotebook/db/          ← Tiki database (41K products)
└── archived/                  ← Old code (can ignore)
```

---

## For Thesis Reviewers

### Key Files to Review:
1. **`ColabNotebook/V4/final/colab_p3_vanna_zero.py`** - Best pipeline
2. **`ColabNotebook/data/eval_300.jsonl`** - Test queries
3. **`ColabNotebook/db/tiki.sqlite`** - Database
4. **`README.md`** - Project overview

### Performance Highlights:
- **300 test queries** (100 simple, 100 medium, 100 complex)
- **76.3% execution accuracy** on P3 Vanna AI
- **Fixed critical bug**: Escape sequence handling (EM=1, EX=0 → EX=1)
- **Production-ready**: All code tested in Google Colab

---

## Requirements

### For P1 & P2:
- Google Colab (free tier works)
- GPU runtime (T4 recommended)

### For P3 (Recommended):
- Google Colab
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- ~$2-5 for 300 query evaluation

---

## Documentation

All documentation is in README files:
- `/code/README.md` - Main overview
- `/code/CLEANUP_COMPLETE.md` - Organization summary
- `/code/PROJECT_STRUCTURE.md` - File structure
- `/ColabNotebook/V4/final/README.md` - Pipeline details

---

## Common Questions

**Q: Which pipeline should I use?**  
A: P3 Vanna AI (`colab_p3_vanna_zero.py`) - highest accuracy (76.3% EX)

**Q: Do I need to install anything?**  
A: No, everything runs in Google Colab

**Q: How long does evaluation take?**  
A: ~10 minutes for 300 queries on Colab GPU

**Q: What if I don't have OpenAI API key?**  
A: Use P1 mT5 (`colab_p1_mt5_zero.py`) - no API key needed, 59% EX

**Q: Where are the results?**  
A: Auto-saved to Google Drive: `/content/drive/MyDrive/vn2sql/logs/`

**Q: Can I run this locally?**  
A: Yes, but Colab GPU is faster and easier

---

## Expected Results (P3 Vanna AI)

```
OVERALL PERFORMANCE:
   Total Queries:        300
   Overall EM:           0.430 (43.0%)
   Overall EX:           0.763 (76.3%)
   Overall Success:      0.870 (87.0%)

Complexity Breakdown:
   Simple  (100): EM=38%, EX=81%
   Medium  (100): EM=29%, EX=84%
   Complex (100): EM=62%, EX=64%
```

---

## Need Help?

1. Check `/code/README.md` for detailed instructions
2. Review `/ColabNotebook/V4/final/README.md` for pipeline specifics
3. Check code comments - comprehensive inline documentation

---

**TL;DR**: Upload `ColabNotebook/V4/final/colab_p3_vanna_zero.py` to Google Colab, add OpenAI API key, run all cells. Get 76.3% execution accuracy on 300 Vietnamese NL2SQL queries. Done.
