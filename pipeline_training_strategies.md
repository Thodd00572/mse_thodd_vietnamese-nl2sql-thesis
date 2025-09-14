# Training Strategies and Comparative Analysis: Pipeline 1 vs Pipeline 2

## Training Strategy Overview

### Pipeline 1: End-to-End Vietnamese NL2SQL
**Architecture**: Vietnamese Query → PhoBERT-SQL → SQL Query

#### Training Approach
1. **Direct Mapping Strategy**
   - Train PhoBERT model directly on Vietnamese NL2SQL pairs
   - 5,000 training samples generated with Vietnamese linguistic patterns
   - Single-step learning: Vietnamese text → SQL output

2. **Vietnamese-Specific Training Data**
   ```
   Vietnamese Query: "tìm áo thun nam giá dưới 500k"
   Target SQL: SELECT * FROM products WHERE name LIKE '%áo thun%' AND name LIKE '%nam%' AND price < 500000
   ```

3. **Linguistic Pattern Training**
   - Vietnamese diacritics handling (á, à, ả, ã, ạ)
   - Compound word recognition ("áo thun", "giày thể thao")
   - Vietnamese price expressions ("500k", "1 triệu")
   - Gender specifications ("nam", "nữ")

4. **Model Architecture**
   - Base: PhoBERT (Vietnamese BERT)
   - Custom head: SQL generation layer
   - Fine-tuning on e-commerce domain vocabulary

### Pipeline 2: Hybrid Translation Approach
**Architecture**: Vietnamese Query → PhoBERT (Vi→En) → SQLCoder (En→SQL) → SQL Query

#### Training Approach
1. **Two-Stage Training Strategy**
   - Stage 1: Vietnamese-to-English translation using Helsinki-NLP/opus-mt-vi-en
   - Stage 2: English-to-SQL using defog/sqlcoder-7b-2

2. **Translation-First Training**
   ```
   Vietnamese: "tìm áo thun nam giá dưới 500k"
   English: "find men's t-shirts under 500k price"
   SQL: SELECT * FROM products WHERE name LIKE '%t-shirt%' AND name LIKE '%men%' AND price < 500000
   ```

3. **Cascaded Learning**
   - Pre-trained Vietnamese-English translator
   - Pre-trained English-SQL generator
   - No end-to-end fine-tuning

## Comparative Analysis Framework

### Why Compare These Two Pipelines?

#### 1. **Architectural Philosophy Comparison**
- **Pipeline 1**: Direct processing minimizes information loss
- **Pipeline 2**: Leverages existing pre-trained models for broader coverage

#### 2. **Language Processing Approach**
- **Pipeline 1**: Native Vietnamese understanding
- **Pipeline 2**: Cross-lingual transfer learning

#### 3. **Training Complexity**
- **Pipeline 1**: Requires domain-specific Vietnamese NL2SQL training data
- **Pipeline 2**: Combines existing models without custom training

## Expected Benefits

### Pipeline 1 Advantages
1. **Linguistic Preservation**
   - Maintains Vietnamese semantic nuances
   - Better handling of Vietnamese-specific expressions
   - Direct understanding of Vietnamese e-commerce terminology

2. **Reduced Error Propagation**
   - Single inference step eliminates translation errors
   - No intermediate representation loss
   - Faster execution time

3. **Domain Optimization**
   - Trained specifically on Vietnamese e-commerce queries
   - Better understanding of Vietnamese product categories
   - Optimized for Tiki database schema

### Pipeline 2 Advantages
1. **Model Reusability**
   - Leverages existing high-quality translation models
   - Benefits from large-scale pre-training
   - No need for extensive Vietnamese NL2SQL training data

2. **Modularity**
   - Independent optimization of translation and SQL generation
   - Easy to replace individual components
   - Better error diagnosis and debugging

3. **Broader Coverage**
   - Translation model trained on diverse Vietnamese text
   - SQL generator benefits from extensive English NL2SQL training

## Expected Tradeoffs

### Performance Tradeoffs

#### Accuracy
- **Pipeline 1**: Higher accuracy on Vietnamese-specific queries
- **Pipeline 2**: Better generalization but potential translation errors

#### Latency
- **Pipeline 1**: ~850ms (single model inference)
- **Pipeline 2**: ~1,250ms (dual model inference)

#### Resource Usage
- **Pipeline 1**: 4.2GB GPU memory (single model)
- **Pipeline 2**: 6.8GB GPU memory (dual models)

### Development Tradeoffs

#### Training Requirements
- **Pipeline 1**: Requires 5,000+ Vietnamese NL2SQL pairs
- **Pipeline 2**: Uses existing pre-trained models

#### Maintenance
- **Pipeline 1**: Single model to maintain and update
- **Pipeline 2**: Two models requiring separate maintenance

#### Scalability
- **Pipeline 1**: Requires retraining for new domains
- **Pipeline 2**: More adaptable to new domains through translation

## Research Implications

### 1. **Cross-lingual NL2SQL Effectiveness**
Comparing direct vs. translation-based approaches provides insights into:
- Information preservation in cross-lingual tasks
- Error propagation in cascaded systems
- Optimal architecture for low-resource languages

### 2. **Vietnamese Language Processing**
- Effectiveness of Vietnamese-specific model training
- Impact of Vietnamese linguistic features on SQL generation
- Comparison with English-centric approaches

### 3. **E-commerce Domain Adaptation**
- Domain-specific vs. general-purpose model performance
- Vietnamese e-commerce terminology handling
- Product search query complexity analysis

## Evaluation Metrics

### Accuracy Metrics
- **Execution Accuracy (EX)**: Percentage of queries producing correct results
- **Exact Match (EM)**: Percentage of queries generating identical SQL

### Efficiency Metrics
- **Average Latency**: Processing time per query
- **Resource Utilization**: GPU memory and CPU usage

### Error Analysis
- **Tonal/Accent Errors**: Vietnamese diacritic handling
- **Compound Word Errors**: Multi-word expression processing
- **SQL Syntax Errors**: Generated query validity
- **Schema Logic Errors**: Correct table relationships

## Expected Research Outcomes

### Hypothesis
Pipeline 1 (End-to-End Vietnamese) will outperform Pipeline 2 (Hybrid Approach) in:
- Execution accuracy (72% vs 68%)
- Processing speed (850ms vs 1,250ms)
- Vietnamese linguistic pattern handling

### Contributions to Field
1. **Benchmark for Vietnamese NL2SQL**: First comprehensive comparison
2. **Cross-lingual Architecture Analysis**: Direct vs. cascaded approaches
3. **Low-resource Language Insights**: Effective strategies for Vietnamese NLP
4. **E-commerce Domain Adaptation**: Practical implementation guidelines

This comparative study provides crucial insights for developing effective NL2SQL systems for Vietnamese and other low-resource languages, with practical implications for e-commerce search applications.
