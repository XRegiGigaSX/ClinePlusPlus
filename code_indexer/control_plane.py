# code_indexer/control_plane.py
"""
Implements the SQLite control plane for repository registration and tracking.
"""
import sqlite3
from code_indexer.config import SQLITE_PATH
from datetime import datetime

CONTROL_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS indexed_repositories (
    repo_id TEXT PRIMARY KEY,
    repo_name TEXT NOT NULL,
    repo_path TEXT UNIQUE NOT NULL,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    postgres_table_name TEXT NOT NULL,
    chroma_collection_name TEXT NOT NULL,
    neo4j_graph_name TEXT NOT NULL
);
"""

def init_control_plane():
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute(CONTROL_TABLE_SQL)
    conn.commit()
    conn.close()

def register_repository(repo_id, repo_name, repo_path,
                        postgres_table_name, chroma_collection_name, neo4j_graph_name):
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute(CONTROL_TABLE_SQL)  # Ensure table exists
    c.execute("""
        INSERT INTO indexed_repositories (
            repo_id, repo_name, repo_path, indexed_at,
            postgres_table_name, chroma_collection_name, neo4j_graph_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        repo_id, repo_name, repo_path, datetime.now(),
        postgres_table_name, chroma_collection_name, neo4j_graph_name
    ))
    conn.commit()
    conn.close()

def list_registered_repositories():
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute(CONTROL_TABLE_SQL)  # Ensure table exists
    c.execute("SELECT * FROM indexed_repositories")
    rows = c.fetchall()
    conn.close()
    return rows

def get_repository_by_path(repo_path):
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute(CONTROL_TABLE_SQL)  # Ensure table exists
    c.execute("SELECT * FROM indexed_repositories WHERE repo_path = ?", (repo_path,))
    row = c.fetchone()
    conn.close()
    return row

def clear_all_repositories():
    """Delete all entries from the indexed_repositories table."""
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute(CONTROL_TABLE_SQL)  # Ensure table exists
    c.execute("DELETE FROM indexed_repositories")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_control_plane()
    print("Control plane initialized.")
