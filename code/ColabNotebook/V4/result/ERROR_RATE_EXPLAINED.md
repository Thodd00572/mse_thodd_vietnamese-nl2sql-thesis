# Error Rate Measurement: Quick Reference Guide
**For Vietnamese NL2SQL Pipeline Evaluation**

---

## 🎯 Quick Definition

**Error Rate** = Percentage of queries where the pipeline **fails to produce valid, executable SQL**

```
Error Rate = (Failed Queries / Total Queries) × 100%
```

---

## ✅ What Counts as an ERROR

A query is marked as **ERROR** (contributes to error rate) when:

1. ❌ **No SQL Output**
   - Model returns empty string or null
   - Pipeline fails to generate any response
   - RAG retrieval returns no results (Vanna AI case)

2. ❌ **Invalid SQL Syntax**
   - Generated text cannot be parsed as SQL
   - Missing required SQL keywords (SELECT, FROM, etc.)
   - Malformed SQL structure

3. ❌ **Database Execution Error**
   - SQL causes runtime error when executed
   - Table/column doesn't exist
   - Type mismatch errors
   - Constraint violations

4. ❌ **System Crash or Timeout**
   - Pipeline crashes during processing
   - Generation takes too long (timeout)
   - Out of memory errors
   - API connection failures

5. ❌ **Non-SQL Output**
   - Model generates text that isn't SQL
   - Returns explanation instead of query
   - Hallucinated or corrupted output

---

## ⚠️ What DOES NOT Count as an ERROR

These do NOT contribute to error rate (even though they're wrong):

1. ✅ **Valid SQL with Wrong Results**
   ```sql
   Query: "Tìm sản phẩm giá dưới 500k"
   Generated: "SELECT * FROM products;"  -- Valid SQL, but ignores price filter
   Error Rate: NO (0% contribution)
   Execution Accuracy: NO (wrong results)
   ```

2. ✅ **Correct Syntax but Wrong Logic**
   ```sql
   Query: "Hiển thị top 10 bán chạy"
   Generated: "SELECT * FROM products ORDER BY price DESC LIMIT 10;"  -- Valid, wrong sort
   Error Rate: NO
   Execution Accuracy: NO
   ```

3. ✅ **Missing Filters but Executable**
   ```sql
   Query: "Sản phẩm Nike có rating > 4"
   Generated: "SELECT * FROM products;"  -- Executes but missing conditions
   Error Rate: NO
   Execution Accuracy: NO
   ```

4. ✅ **Wrong Table but Valid SQL**
   ```sql
   Query: "Xem danh sách sản phẩm"
   Generated: "SELECT * FROM brands;"  -- Valid SQL, wrong table
   Error Rate: NO
   Execution Accuracy: NO
   ```

---

## 📊 Real Examples from Evaluation

### Example 1: P2 SQLCoder (0% Error Rate)

**Query**: "Tìm áo thun có giá từ 200k đến 500k"

**Generated SQL**:
```sql
SELECT * FROM products;
```

**Analysis**:
- ✅ Valid SQL syntax
- ✅ Executes without error
- ❌ Ignores "áo thun" (t-shirt) filter
- ❌ Ignores price range 200k-500k
- **Error Rate**: 0% (contributes nothing)
- **Execution Accuracy**: 0% (wrong results)

**Conclusion**: Low error rate ≠ good performance

---

### Example 2: P3 Vanna AI (13% Error Rate)

**Query**: "Phân tích xu hướng giá laptop theo thương hiệu"

**Generated SQL**:
```
None (RAG retrieval failed - no similar examples found)
```

**Analysis**:
- ❌ No SQL output
- ❌ Cannot execute
- **Error Rate**: 100% (contributes to 13% overall)
- **Execution Accuracy**: N/A (can't measure)

**Conclusion**: High error rate for complex queries, but better than wrong SQL

---

### Example 3: P1 mT5 (1.3% Error Rate)

**Query**: "Hiển thị top 20 sản phẩm bán chạy có đánh giá trên 4.5 sao"

**Generated SQL**:
```sql
SELECT p.name, pr.quantity_sold, rv.rating_average 
FROM products p 
JOIN JOIN product_pricing pr ON p.product_id = pr.product_id  -- DOUBLE JOIN ERROR
WHERE rv.rating_average > 4.5 
ORDER BY pr.quantity_sold DESC 
LIMIT 20;
```

**Analysis**:
- ❌ Invalid syntax (double JOIN keyword)
- ❌ Parser error
- **Error Rate**: 100% (contributes to 1.3% overall)
- **Execution Accuracy**: N/A

**Conclusion**: Rare syntax errors when model hallucinates

---

## 🔬 Error Rate vs Execution Accuracy

### The Critical Distinction:

| Metric | What It Measures | Good/Bad |
|--------|------------------|----------|
| **Low Error Rate** | System produces SQL reliably | ✅ Good for stability |
| **High Execution Accuracy** | SQL returns correct results | ✅ Good for users |

### Four Possible Scenarios:

#### Scenario 1: Low Error + Low Accuracy (P2 SQLCoder)
```
Error Rate: 0% ✅
Execution Accuracy: 22.3% ❌
Result: Always produces SQL, usually wrong
User Experience: Confidently wrong (worst case)
```

#### Scenario 2: Low Error + High Accuracy (Ideal - not achieved)
```
Error Rate: 0% ✅
Execution Accuracy: 90%+ ✅
Result: Always produces correct SQL
User Experience: Perfect reliability (best case)
```

#### Scenario 3: High Error + Low Accuracy (Worst)
```
Error Rate: 50% ❌
Execution Accuracy: 20% ❌
Result: Often fails, and when succeeds, usually wrong
User Experience: Unreliable and inaccurate
```

#### Scenario 4: Moderate Error + High Accuracy (P3 Vanna AI)
```
Error Rate: 13% ⚠️
Execution Accuracy: 76.3% ✅
Conditional Accuracy (when succeeds): 87.7% ✅
Result: Sometimes fails, but when succeeds, usually correct
User Experience: Reliable enough with high accuracy
```

---

## 🎓 Why This Matters for Research

### Common Misconception:
> "0% error rate means the system works perfectly"

### Reality:
> "0% error rate only means the system always produces *something*, not that it produces the *right thing*"

### Thesis Contribution:
**First Vietnamese NL2SQL study to distinguish between:**
1. **Technical Reliability** (Error Rate) - Can we generate SQL?
2. **Functional Correctness** (Execution Accuracy) - Is the SQL right?
3. **Conditional Performance** (EX when succeeds) - Quality when working

---

## 📈 Practical Implications

### For Production Systems:

**Option A: Prioritize Low Error Rate (P2 approach)**
- Pro: Never fails
- Con: Often wrong
- Use when: Any SQL output needed (logging, analytics)

**Option B: Prioritize High Accuracy (P3 approach)**
- Pro: Usually correct when succeeds
- Con: Sometimes fails
- Use when: Correctness critical (e-commerce, finance)

**Option C: Hybrid System (Recommended)**
```python
def generate_sql(query: str) -> str:
    # Try high-accuracy pipeline first
    result = vanna_ai(query)
    if result.success:
        return result.sql  # 87.7% likely correct
    
    # Fallback to fast pipeline
    result = mt5_fallback(query)
    return result.sql  # Better than nothing
```

**Result**: 99.8% reliability (no errors) with 80.6% useful results

---

## 🔑 Key Takeaways

1. ✅ **Error Rate measures system stability**, not accuracy
2. ✅ **0% error can coexist with 20% accuracy** (P2 SQLCoder proves this)
3. ✅ **13% error with 87.7% conditional accuracy** is often better than 0% error with 22.3% accuracy
4. ✅ **Production systems should measure BOTH** error rate and execution accuracy
5. ✅ **Graceful failures > Confidently wrong answers** for user trust

---

## 📊 Summary Table

| Pipeline | Error Rate | What Errors Mean | EX | Recommendation |
|----------|-----------|------------------|-----|----------------|
| **P2 SQLCoder** | 0.0% | N/A (never fails) | 22.3% | Use for logging/analytics |
| **P1 mT5** | 1.3% | Rare syntax errors | 32.7% | Use for speed-critical apps |
| **P3 Vanna AI** | 13.0% | RAG retrieval failures | 76.3% | **Use for accuracy-critical apps** ⭐ |

---

## 🎯 Bottom Line

**Error Rate tells you:** "How often does the system crash or fail to generate SQL?"

**Execution Accuracy tells you:** "How often does the system give you the right answer?"

**For Vietnamese NL2SQL, we care MORE about giving right answers WHEN we answer, than answering every time with wrong results.**

**That's why P3 Vanna AI with 13% error rate is BETTER than P2 SQLCoder with 0% error rate.**

---

**Last Updated**: October 1, 2025, 19:30 ICT  
**Related Documents**: 
- `ERROR_RATE_ANALYSIS.md` - Full analysis
- `EVALUATION_SUMMARY_20251001.md` - Complete results
- Analysis Dashboard: http://localhost:3000/analysis
