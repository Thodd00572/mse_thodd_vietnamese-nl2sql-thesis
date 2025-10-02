#!/usr/bin/env python3
"""
Create comprehensive development/validation dataset for Vietnamese NL2SQL model
Uses the remaining 30% of queries not used in training for proper validation
"""

import json
import random
from pathlib import Path

def create_dev_dataset():
    """Create a comprehensive development dataset for validation"""
    
    # Input and output paths
    input_file = Path("/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/frontend/public/data/sample_queries_complete.json")
    train_file = Path("/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/ColabNotebook/V3/train.jsonl")
    output_file = Path("/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/ColabNotebook/V3/dev.jsonl")
    
    print(f"📖 Reading from: {input_file}")
    
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Read existing training data to avoid overlap
    print(f"📖 Reading existing training data from: {train_file}")
    train_queries_vn = set()
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    query = json.loads(line.strip())
                    if query['origin'] == 'human':  # Only check human queries to avoid synthetic overlaps
                        train_queries_vn.add(query['vn'])
    
    print(f"📊 Found {len(train_queries_vn)} human queries in training set")
    
    # Strategy: Use remaining 30% of queries for development (not used in training)
    dev_queries = []
    
    # Process each complexity level
    for complexity_level in ['simple', 'medium', 'complex']:
        if complexity_level in data:
            queries = data[complexity_level]
            
            # Find queries not used in training
            available_queries = []
            for query in queries:
                if query['vietnamese'] not in train_queries_vn:
                    available_queries.append(query)
            
            # Take up to 30 queries per complexity level for dev set
            num_dev = min(30, len(available_queries))
            
            # Shuffle and select dev queries
            random.seed(42)  # For reproducibility
            selected_queries = random.sample(available_queries, num_dev)
            
            for query in selected_queries:
                dev_queries.append({
                    'vn': query['vietnamese'],
                    'sql': query['sql'],
                    'complexity': query['complexity'],
                    'origin': 'human'
                })
            
            print(f"📊 {complexity_level}: Selected {num_dev} queries for dev set (from {len(available_queries)} available)")
    
    # Add some additional validation-specific queries to test edge cases
    validation_specific_queries = [
        # Edge cases for validation
        {"vn": "Sản phẩm không có giá", "sql": "SELECT * FROM products WHERE price IS NULL LIMIT 10;", "complexity": "simple", "origin": "validation"},
        {"vn": "Sản phẩm có mô tả trống", "sql": "SELECT * FROM products WHERE description IS NULL OR description = '' LIMIT 10;", "complexity": "simple", "origin": "validation"},
        {"vn": "Thương hiệu có nhiều sản phẩm nhất", "sql": "SELECT brand, COUNT(*) as product_count FROM products GROUP BY brand ORDER BY product_count DESC LIMIT 1;", "complexity": "medium", "origin": "validation"},
        {"vn": "Danh mục có ít sản phẩm nhất", "sql": "SELECT category, COUNT(*) as product_count FROM products GROUP BY category ORDER BY product_count ASC LIMIT 1;", "complexity": "medium", "origin": "validation"},
        {"vn": "Sản phẩm có tên dài nhất", "sql": "SELECT name, LENGTH(name) as name_length FROM products ORDER BY name_length DESC LIMIT 1;", "complexity": "medium", "origin": "validation"},
        
        # Boundary testing
        {"vn": "Sản phẩm giá bằng 0", "sql": "SELECT * FROM products WHERE price = 0 LIMIT 10;", "complexity": "simple", "origin": "validation"},
        {"vn": "Sản phẩm có ID lớn nhất", "sql": "SELECT * FROM products ORDER BY id DESC LIMIT 1;", "complexity": "simple", "origin": "validation"},
        {"vn": "Tổng giá trị tất cả sản phẩm", "sql": "SELECT SUM(price) as total_value FROM products;", "complexity": "medium", "origin": "validation"},
        
        # Vietnamese language variations for validation
        {"vn": "Tìm kiếm sản phẩm có từ 'cao cấp'", "sql": "SELECT * FROM products WHERE description LIKE '%cao cấp%' LIMIT 10;", "complexity": "simple", "origin": "validation"},
        {"vn": "Sản phẩm có tên chứa số", "sql": "SELECT * FROM products WHERE name REGEXP '[0-9]' LIMIT 10;", "complexity": "medium", "origin": "validation"},
    ]
    
    # Add validation-specific queries
    dev_queries.extend(validation_specific_queries)
    
    # Shuffle the final dev set
    random.shuffle(dev_queries)
    
    print(f"📊 Total dev queries: {len(dev_queries)}")
    
    # Count by complexity and origin
    complexity_counts = {}
    origin_counts = {}
    
    for query in dev_queries:
        complexity = query['complexity']
        origin = query['origin']
        
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
    
    print("📈 Breakdown by complexity:")
    for complexity, count in complexity_counts.items():
        print(f"   {complexity}: {count} queries")
    
    print("📈 Breakdown by origin:")
    for origin, count in origin_counts.items():
        print(f"   {origin}: {count} queries")
    
    # Write to JSONL format
    print(f"💾 Writing to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in dev_queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully created dev.jsonl with {len(dev_queries)} queries")
    
    # Verify no overlap with training data
    overlap_count = 0
    for query in dev_queries:
        if query['vn'] in train_queries_vn:
            overlap_count += 1
    
    print(f"🔍 Data leakage check: {overlap_count} overlapping queries (should be 0 for human queries)")
    
    # Verify the output
    print("\n🔍 Sample entries from dev.jsonl:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5:  # Show first 5 entries
                query = json.loads(line.strip())
                print(f"   {i+1}. VN: {query['vn']}")
                print(f"      SQL: {query['sql'][:80]}{'...' if len(query['sql']) > 80 else ''}")
                print(f"      Complexity: {query['complexity']}, Origin: {query['origin']}")
                print()
    
    return output_file, len(dev_queries), complexity_counts, origin_counts

if __name__ == "__main__":
    output_file, total_queries, complexity_counts, origin_counts = create_dev_dataset()
    
    print(f"\n🎯 Summary:")
    print(f"   📄 Output file: {output_file}")
    print(f"   📊 Total queries: {total_queries}")
    print(f"   📈 Complexity distribution: {complexity_counts}")
    print(f"   📈 Origin distribution: {origin_counts}")
    print(f"\n✅ Development dataset ready for validation during training!")
    print(f"💡 This dataset will be used for:")
    print(f"   - Validation during training (eval_dataset)")
    print(f"   - Early stopping based on validation metrics")
    print(f"   - Hyperparameter tuning and model selection")
    print(f"   - Preventing overfitting to training data")
