import sqlite3
import sys

def get_tables(conn):
    """Get all table names from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall()]

def get_columns(conn, table_name):
    """Get all column names for a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cursor.fetchall()]

def find_duplicates(conn, table_name, key_columns):
    """Find duplicate rows based on specified columns."""
    cursor = conn.cursor()
    key_cols_str = ', '.join(key_columns)
    
    query = f"""
    SELECT {key_cols_str}, COUNT(*) as count
    FROM {table_name}
    GROUP BY {key_cols_str}
    HAVING count > 1
    """
    
    cursor.execute(query)
    return cursor.fetchall()

def remove_duplicates(conn, table_name, key_columns, keep='first'):
    """
    Remove duplicate rows, keeping only one record per duplicate group.
    
    Args:
        conn: Database connection
        table_name: Name of the table
        key_columns: List of columns that define a duplicate
        keep: 'first' keeps lowest rowid, 'last' keeps highest rowid
    """
    cursor = conn.cursor()
    key_cols_str = ', '.join(key_columns)
    
    # Get the primary key or rowid
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns_info = cursor.fetchall()
    pk_col = next((col[1] for col in columns_info if col[5] == 1), 'rowid')
    
    if keep == 'first':
        aggregate = 'MIN'
    elif keep == 'last':
        aggregate = 'MAX'
    else:
        raise ValueError("keep must be 'first' or 'last'")
    
    # Delete duplicates, keeping one record per group
    delete_query = f"""
    DELETE FROM {table_name}
    WHERE {pk_col} NOT IN (
        SELECT {aggregate}({pk_col})
        FROM {table_name}
        GROUP BY {key_cols_str}
    )
    """
    
    cursor.execute(delete_query)
    deleted_count = cursor.rowcount
    conn.commit()
    
    return deleted_count

def main():
    db_name = 'domain_pricing.db'
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_name)
        print(f"Connected to {db_name}\n")
        
        # Get all tables
        tables = get_tables(conn)
        print(f"Tables found: {', '.join(tables)}\n")
        
        # Process each table
        for table in tables:
            print(f"\n--- Processing table: {table} ---")
            columns = get_columns(conn, table)
            print(f"Columns: {', '.join(columns)}")
            
            # Get total count before
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count_before = cursor.fetchone()[0]
            print(f"Total rows before: {count_before}")
            
            # Remove duplicates based on ALL columns (exact duplicate rows)
            # This means every field must match for a row to be considered duplicate
            key_columns = columns
            
            print(f"\nChecking for exact duplicate rows (all columns must match)")
            duplicates = find_duplicates(conn, table, key_columns)
            
            if duplicates:
                print(f"Found {len(duplicates)} duplicate groups")
                
                # Show first few examples
                print("\nFirst 3 duplicate examples:")
                for i, dup in enumerate(duplicates[:3], 1):
                    print(f"\n  Example {i}:")
                    for col, val in zip(key_columns, dup[:-1]):  # Exclude count
                        print(f"    {col}: {val}")
                    print(f"    Occurrences: {dup[-1]}")
                
                # Ask for confirmation
                response = input(f"\nRemove duplicates from '{table}'? (yes/no): ")
                
                if response.lower() in ['yes', 'y']:
                    deleted = remove_duplicates(conn, table, key_columns, keep='first')
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count_after = cursor.fetchone()[0]
                    print(f"\n✓ Deleted {deleted} duplicate rows")
                    print(f"  Before: {count_before} rows")
                    print(f"  After:  {count_after} rows")
                else:
                    print("Skipped removal")
            else:
                print("No duplicates found")
        
        conn.close()
        print("\n✓ Operation completed successfully")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()