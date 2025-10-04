# code_indexer/db_init.py
# Utilities to initialize and test connections to all databases.

import psycopg2
from neo4j import GraphDatabase
import chromadb
import sqlite3
from config import *

def test_postgres():
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        conn.close()
        print('PostgreSQL connection successful.')
    except Exception as e:
        print(f'PostgreSQL connection failed: {e}')

def test_neo4j():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run('RETURN 1')
        print('Neo4j connection successful.')
    except Exception as e:
        print(f'Neo4j connection failed: {e}')

def test_chromadb():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        print('ChromaDB connection successful.')
    except Exception as e:
        print(f'ChromaDB connection failed: {e}')

def test_sqlite():
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.close()
        print('SQLite connection successful.')
    except Exception as e:
        print(f'SQLite connection failed: {e}')

def test_all():
    test_postgres()
    test_neo4j()
    test_chromadb()
    test_sqlite()

if __name__ == '__main__':
    test_all()
