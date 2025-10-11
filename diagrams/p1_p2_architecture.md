# Vietnamese NL2SQL Pipeline Architecture Diagrams

## Pipeline 1: mT5 Zero-Shot Prompting

```mermaid
graph TD
    A[Vietnamese Query] --> B[Schema-Aware Prompting<br/>• Database Schema DDL<br/>• Few-Shot Examples<br/>• Vietnamese SQL Rules]
    B --> C[mT5 Multilingual Model<br/>google/mt5-base]
    C --> D[SQL Post-Processing<br/>• Extract SQL<br/>• Normalize Format<br/>• Add LIMIT Clauses]
    D --> E[Tiki E-commerce Dataset]
    E --> F[Search Result]
    
    style A fill:#B3D9FF,stroke:#0066CC,stroke-width:2px
    style B fill:#FFD9B3,stroke:#CC6600,stroke-width:2px
    style C fill:#D9B3FF,stroke:#6600CC,stroke-width:2px
    style D fill:#FFFFB3,stroke:#CCCC00,stroke-width:2px
    style E fill:#FFB3B3,stroke:#CC0000,stroke-width:2px
    style F fill:#B3FFB3,stroke:#00CC00,stroke-width:2px
```

## Pipeline 2: SQLCoder Zero-Shot

```mermaid
graph TD
    A[Vietnamese Query] --> B[Schema-Aware Prompting<br/>• Database Schema DDL<br/>• SQLCoder Format Examples<br/>• Vietnamese SQL Rules]
    B --> C[SQLCoder Model<br/>defog/sqlcoder-7b-2<br/>FP16 Precision]
    C --> D[SQL Cleaning & Processing<br/>• Fix LIKE Patterns<br/>• Normalize Tables/Columns<br/>• Remove SQLCoder Artifacts]
    D --> E[Tiki E-commerce Dataset]
    E --> F[Search Result]
    
    style A fill:#B3D9FF,stroke:#0066CC,stroke-width:2px
    style B fill:#FFD9B3,stroke:#CC6600,stroke-width:2px
    style C fill:#D9B3FF,stroke:#6600CC,stroke-width:2px
    style D fill:#FFFFB3,stroke:#CCCC00,stroke-width:2px
    style E fill:#FFB3B3,stroke:#CC0000,stroke-width:2px
    style F fill:#B3FFB3,stroke:#00CC00,stroke-width:2px
```

## Pipeline 3: Vanna AI RAG (Reference)

```mermaid
graph TD
    A[Vietnamese Query] --> B[Vanna AI RAG Training<br/>• Database Schema DDL<br/>• 98 Training Pairs<br/>• Vietnamese SQL Docs]
    B --> C[ChromaDB Vector Store<br/>Semantic Retrieval]
    C --> D[OpenAI API<br/>GPT-4o-mini Generation]
    D --> E[SQL Query Processing]
    E --> F[Tiki E-commerce Dataset]
    F --> G[Search Result]
    
    style A fill:#B3D9FF,stroke:#0066CC,stroke-width:2px
    style B fill:#FFD9B3,stroke:#CC6600,stroke-width:2px
    style C fill:#D9B3FF,stroke:#6600CC,stroke-width:2px
    style D fill:#B3FFB3,stroke:#00CC00,stroke-width:2px
    style E fill:#FFFFB3,stroke:#CCCC00,stroke-width:2px
    style F fill:#FFB3B3,stroke:#CC0000,stroke-width:2px
    style G fill:#B3FFB3,stroke:#00CC00,stroke-width:2px
```

## Key Differences Summary

| Component | P1 mT5 | P2 SQLCoder | P3 Vanna AI |
|-----------|--------|-------------|-------------|
| **Model** | google/mt5-base (580M params) | defog/sqlcoder-7b-2 (7B params) | OpenAI GPT-4o-mini |
| **Approach** | Direct prompting | Direct prompting | RAG retrieval + generation |
| **Context** | Static few-shot examples | Static few-shot examples | Dynamic ChromaDB retrieval |
| **Latency** | 0.30s | 1.76s | ~0.50s |
| **GPU Memory** | 2.5 GB | 13.8 GB | API-based (no local GPU) |
| **EM Performance** | 16% | 18% | ~27% (with fixes) |
| **EX Performance** | 33% | 22% | ~57% (with fixes) |
| **Processing** | Minimal post-processing | Heavy SQL cleaning | Rule-based fallbacks |

## Architecture Patterns

### P1 & P2 (Zero-Shot Prompting)
- **Single-stage generation**: Query → Prompt → Model → SQL
- **Static context**: Same examples for all queries
- **Local inference**: GPU-based generation
- **Trade-off**: Speed vs accuracy (mT5 faster, SQLCoder more accurate)

### P3 (RAG-based)
- **Two-stage generation**: Query → Retrieval → Context + Generation → SQL
- **Dynamic context**: Retrieved similar examples per query
- **Hybrid approach**: Vector store + OpenAI API
- **Trade-off**: Better context awareness but API dependency
