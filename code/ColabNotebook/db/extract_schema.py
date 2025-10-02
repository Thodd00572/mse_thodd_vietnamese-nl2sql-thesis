#!/usr/bin/env python3
"""
Extract SQLite database schema and save as JSON
Generates comprehensive schema information for the Tiki database
"""

import sqlite3
import json
import os
from pathlib import Path

def extract_table_info(cursor, table_name):
    """Extract detailed information about a table"""
    # Get column information
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = []
    for row in cursor.fetchall():
        cid, name, data_type, not_null, default_value, pk = row
        columns.append({
            "column_id": cid,
            "name": name,
            "data_type": data_type,
            "not_null": bool(not_null),
            "default_value": default_value,
            "primary_key": bool(pk)
        })
    
    # Get foreign key information
    cursor.execute(f"PRAGMA foreign_key_list({table_name})")
    foreign_keys = []
    for row in cursor.fetchall():
        id, seq, table, from_col, to_col, on_update, on_delete, match = row
        foreign_keys.append({
            "id": id,
            "sequence": seq,
            "referenced_table": table,
            "from_column": from_col,
            "to_column": to_col,
            "on_update": on_update,
            "on_delete": on_delete,
            "match": match
        })
    
    # Get index information
    cursor.execute(f"PRAGMA index_list({table_name})")
    indexes = []
    for row in cursor.fetchall():
        seq, name, unique, origin, partial = row
        cursor.execute(f"PRAGMA index_info({name})")
        index_columns = []
        for idx_row in cursor.fetchall():
            seqno, cid, col_name = idx_row
            index_columns.append({
                "sequence": seqno,
                "column_id": cid,
                "column_name": col_name
            })
        
        indexes.append({
            "sequence": seq,
            "name": name,
            "unique": bool(unique),
            "origin": origin,
            "partial": bool(partial),
            "columns": index_columns
        })
    
    return {
        "name": table_name,
        "columns": columns,
        "foreign_keys": foreign_keys,
        "indexes": indexes
    }

def extract_view_info(cursor, view_name):
    """Extract information about a view"""
    # Get the view definition
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='view' AND name=?", (view_name,))
    result = cursor.fetchone()
    view_sql = result[0] if result else None
    
    # Get column information by querying the view structure
    cursor.execute(f"PRAGMA table_info({view_name})")
    columns = []
    for row in cursor.fetchall():
        cid, name, data_type, not_null, default_value, pk = row
        columns.append({
            "column_id": cid,
            "name": name,
            "data_type": data_type,
            "not_null": bool(not_null),
            "default_value": default_value,
            "primary_key": bool(pk)
        })
    
    return {
        "name": view_name,
        "type": "view",
        "columns": columns,
        "definition": view_sql
    }

def get_sample_data(cursor, table_name, limit=5):
    """Get sample data from a table"""
    try:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        rows = cursor.fetchall()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        column_names = [row[1] for row in cursor.fetchall()]
        
        # Convert rows to dictionaries
        sample_data = []
        for row in rows:
            sample_data.append(dict(zip(column_names, row)))
        
        return sample_data
    except Exception as e:
        return f"Error getting sample data: {str(e)}"

def extract_database_schema(db_path):
    """Extract complete database schema"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    schema = {
        "database_file": str(db_path),
        "extraction_timestamp": None,
        "tables": [],
        "views": [],
        "statistics": {}
    }
    
    # Get current timestamp
    import datetime
    schema["extraction_timestamp"] = datetime.datetime.now().isoformat()
    
    # Get all tables (excluding system tables)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    table_names = [row[0] for row in cursor.fetchall()]
    
    # Extract information for each table
    for table_name in table_names:
        print(f"Extracting schema for table: {table_name}")
        table_info = extract_table_info(cursor, table_name)
        
        # Add row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        table_info["row_count"] = cursor.fetchone()[0]
        
        # Add sample data
        table_info["sample_data"] = get_sample_data(cursor, table_name)
        
        schema["tables"].append(table_info)
    
    # Get all views
    cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
    view_names = [row[0] for row in cursor.fetchall()]
    
    # Extract information for each view
    for view_name in view_names:
        print(f"Extracting schema for view: {view_name}")
        view_info = extract_view_info(cursor, view_name)
        
        # Add sample data
        view_info["sample_data"] = get_sample_data(cursor, view_name)
        
        schema["views"].append(view_info)
    
    # Add database statistics
    schema["statistics"] = {
        "total_tables": len(table_names),
        "total_views": len(view_names),
        "table_names": table_names,
        "view_names": view_names
    }
    
    # Add total row counts
    total_rows = sum(table["row_count"] for table in schema["tables"])
    schema["statistics"]["total_rows"] = total_rows
    
    conn.close()
    return schema

def main():
    # Database path
    db_path = Path(__file__).parent / "tiki.sqlite"
    
    if not db_path.exists():
        print(f"❌ Database file not found: {db_path}")
        return
    
    print(f"📊 Extracting schema from: {db_path}")
    
    # Extract schema
    schema = extract_database_schema(db_path)
    
    # Save to JSON file
    output_path = Path(__file__).parent / "tiki_schema.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Schema saved to: {output_path}")
    print(f"📈 Database Statistics:")
    print(f"   - Tables: {schema['statistics']['total_tables']}")
    print(f"   - Views: {schema['statistics']['total_views']}")
    print(f"   - Total Rows: {schema['statistics']['total_rows']:,}")
    
    # Print table summary
    print(f"\n📋 Table Summary:")
    for table in schema["tables"]:
        print(f"   - {table['name']}: {table['row_count']:,} rows, {len(table['columns'])} columns")
    
    if schema["views"]:
        print(f"\n👁️ Views:")
        for view in schema["views"]:
            print(f"   - {view['name']}: {len(view['columns'])} columns")

if __name__ == "__main__":
    main()
