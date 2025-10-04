# code_indexer/config.py
# Centralized configuration for database connections and paths.

import os

# PostgreSQL connection settings (edit as needed)
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
POSTGRES_DB = os.getenv('POSTGRES_DB', 'code_indexer')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'rayyan2008')

# Neo4j connection settings (edit as needed)
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'rayyan2008')

# ChromaDB storage path
CHROMA_PATH = os.getenv('CHROMA_PATH', os.path.join(os.path.dirname(__file__), 'chroma_storage'))

# SQLite control plane path
SQLITE_PATH = os.getenv('SQLITE_PATH', os.path.join(os.path.dirname(__file__), 'repository_registry.db'))
