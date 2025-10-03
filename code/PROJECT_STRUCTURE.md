# MSE Thesis 2025 - Project Structure

## Clean Directory Tree

```
code/
│
├── README.md                    Main project documentation
├── CLEANUP_COMPLETE.md         Cleanup summary and achievements
├── PROJECT_STRUCTURE.md        This file
│
├── ColabNotebook/              PRODUCTION CODE (Main thesis work)
│   │
│   ├── README.md               Colab project overview
│   ├── ORGANIZATION_SUMMARY.md Organization details
│   ├── CLEANUP_COMPLETE.md     Cleanup summary
│   │
│   ├── V4/                     Version 4 (Final)
│   │   │
│   │   ├── final/              PRODUCTION PIPELINES
│   │   │   │
│   │   │   ├── README.md      Pipeline documentation
│   │   │   │
│   │   │   ├── colab_p1_mt5_zero.py         P1: mT5 (48% EM, 59% EX)
│   │   │   ├── colab_p2_sqlcoder_zero.py    P2: SQLCoder (19% EM, 21% EX)
│   │   │   └── colab_p3_vanna_zero.py       P3: Vanna AI (43% EM, 76.3% EX) [BEST]
│   │   │   │
│   │   │   └── All/            Archive subdirectory
│   │   │       └── archive/
│   │   │           ├── documentation/  Historical markdown docs
│   │   │           └── debug_scripts/  Debug utilities
│   │   │
│   │   ├── result/             Evaluation results
│   │   ├── requirements.txt    Python dependencies
│   │   └── docs/               Additional documentation
│   │
│   ├── data/                   EVALUATION DATASETS
│   │   ├── eval_300.jsonl         300 test queries
│   │   ├── train.jsonl            Training examples
│   │   └── ...                    Other data files
│   │
│   ├── db/                     TIKI DATABASE
│   │   └── tiki.sqlite            41,576 products, 5 tables
│   │
│   ├── diagram/                ARCHITECTURE DIAGRAMS
│   │   └── p3-architecture.mmd    P3 Vanna AI diagram
│   │
│   ├── old/                    Original thesis notebooks
│   │
│   └── unused/                 Colab-specific archive (72 items)
│       ├── README.md           Archive explanation
│       ├── phobert_related/       PhoBERT experiments (5 files)
│       ├── old_pipeline_versions/ P3/P4/P5 old versions (12 files)
│       ├── v2_v3_legacy/          V2 and V3 code
│       ├── experimental/          Test scripts (8 files)
│       └── V4_docs/               Historical docs (15 files)
│
├── demo/                       DEMO APPLICATION (for presentations)
│   │
│   ├── README.md               Demo app documentation
│   │
│   ├── backend/                FastAPI server (65 items)
│   ├── frontend/               React/Next.js UI (24 items)
│   └── start_thesis_app.sh     Startup script
│
└── archived/                   NON-PRODUCTION CODE (~650MB)
    │
    ├── phobert_models/         PhoBERT trained models (~650MB)
    │   └── LocalModel_PhoBERT/
    │       ├── phobert_sql_trained.pth (650MB)
    │       └── phobert_sql_trained_tokenizer/
    │
    ├── local_evaluation/       Local evaluation scripts (17 items)
    │   ├── four_model_local_evaluator.py
    │   ├── schema_aligned_evaluator.py
    │   └── ...
    │
    ├── utility_scripts/        Database and setup utilities
    │   ├── explore_tiki_db.py
    │   ├── import_tiki_data.py
    │   ├── local/                 MacMini pipeline
    │   ├── local_models/          Model utilities
    │   └── ...
    │
    └── old_unused_files/       Previously archived code (8 items)
```

---

## Navigation Guide

### For Production Thesis Work
```bash
cd ColabNotebook/V4/final/
# Three production pipelines here
```

### For Evaluation Data
```bash
cd ColabNotebook/data/
# eval_300.jsonl (300 test queries)
```

### For Database
```bash
cd ColabNotebook/db/
# tiki.sqlite (Tiki e-commerce database)
```

### For Architecture Diagrams
```bash
cd ColabNotebook/diagram/
# Mermaid architecture diagrams
```

---

## File Count Summary

| Category | Location | Count | Size | Status |
|----------|----------|-------|------|--------|
| **Production Pipelines** | ColabNotebook/V4/final/ | 3 files | ~200KB | Active |
| **Evaluation Data** | ColabNotebook/data/ | 12 files | ~5MB | Active |
| **Database** | ColabNotebook/db/ | 1 file | ~150MB | Active |
| **Diagrams** | ColabNotebook/diagram/ | 2 files | ~10KB | Active |
| **Colab Archive** | ColabNotebook/unused/ | 72 items | ~2MB | Archived |
| **Demo App** | archived/demo_app/ | 89 items | ~500MB | Archived |
| **PhoBERT Models** | archived/phobert_models/ | 6 items | ~650MB | Archived |
| **Local Evaluation** | archived/local_evaluation/ | 17 items | ~1MB | Archived |
| **Utilities** | archived/utility_scripts/ | 10+ items | ~50KB | Archived |
| **Old Files** | archived/old_unused_files/ | 8 items | ~2MB | Archived |

**Total Active**: ~155MB  
**Total Archived**: ~1.2GB  

---

## Quick Access

### Best Pipeline (Recommended)
```
ColabNotebook/V4/final/colab_p3_vanna_zero.py
```
- 76.3% Execution Accuracy (BEST)
- RAG + GPT-4o approach
- Production-ready

### Evaluation Dataset
```
ColabNotebook/data/eval_300.jsonl
```
- 300 queries (100 simple, 100 medium, 100 complex)

### Database
```
ColabNotebook/db/tiki.sqlite
```
- 41,576 products
- 5 tables (products, brands, categories, pricing, reviews)

---

## Documentation Map

| File | Purpose | Location |
|------|---------|----------|
| Main README | Project overview | `/code/README.md` |
| Cleanup Summary | Organization details | `/code/CLEANUP_COMPLETE.md` |
| Project Structure | This file | `/code/PROJECT_STRUCTURE.md` |
| Colab README | Colab project docs | `/ColabNotebook/README.md` |
| Pipeline Guide | Usage instructions | `/ColabNotebook/V4/final/README.md` |
| Colab Archive | Unused code explanation | `/ColabNotebook/unused/README.md` |
| Archive README | Non-production code | `/archived/README.md` |

---

## For Thesis Submission

### Must Include
- `ColabNotebook/V4/final/` - Production pipelines
- `ColabNotebook/data/` - Evaluation datasets
- `ColabNotebook/db/` - Database
- `ColabNotebook/diagram/` - Architecture
- All README files

### Optional
- `ColabNotebook/unused/` - Experimental approaches
- `archived/` - Historical reference

### Can Omit
- `archived/` - Not core to thesis

---

**Last Updated**: October 1, 2025  
**Status**: Production-ready  
**Total Organization**: 200+ files organized  
**Active Code**: 3 pipelines in `/ColabNotebook/V4/final/`  
