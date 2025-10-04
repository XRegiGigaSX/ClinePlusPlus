from transformers import AutoTokenizer, AutoModel
import torch
def generate_codet5_embedding(text, tokenizer, model, device):
    # Tokenize and get embedding from CodeT5-small
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.encoder(**inputs)
        # Use mean pooling over the sequence output
        last_hidden = outputs.last_hidden_state
        embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()
    return embedding.tolist()
from code_indexer.code_chunker import chunk_repository
# Endpoint to trigger indexing (chunking) for a registered repository
from fastapi import Query, Request
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import psycopg2
from neo4j import GraphDatabase
import chromadb
import sqlite3
from code_indexer.config import *

from code_indexer.control_plane import list_registered_repositories, init_control_plane, register_repository, clear_all_repositories
import re

# Instantiate FastAPI app at the top
app = FastAPI()


@app.post("/api/index_repository")
async def index_repository(request: Request):
    data = await request.json()
    repo_path = data.get('repo_path')
    if not repo_path:
        print("No repo_path provided.")
        return JSONResponse({"error": "repo_path is required"}, status_code=400)
    # Find the Postgres table name for this repo
    rows = list_registered_repositories()
    columns = ["repo_id", "repo_name", "repo_path", "indexed_at", "postgres_table_name", "chroma_collection_name", "neo4j_graph_name"]
    repo = None
    for row in rows:
        r = dict(zip(columns, row))
        if r["repo_path"] == repo_path:
            repo = r
            break
    if not repo:
        print(f"Repository not registered for path: {repo_path}")
        return JSONResponse({"error": "Repository not registered"}, status_code=404)
    table_name = repo["postgres_table_name"]
    print(f"Indexing repository at {repo_path}, using table {table_name}")

    # Call the chunker
    print("[KG] Starting chunking...")
    chunks, error_files = chunk_repository(repo_path)
    print(f"[KG] Chunking complete. Chunks found: {len(chunks)}. Error files: {len(error_files)}")

    # --- Python KG Extraction ---
    print("[KG] Starting Python KG extraction...")
    kg_nodes = []
    kg_rels = []
    file_nodes = {}
    class_nodes = {}
    func_nodes = {}
    def file_node_id(file_path):
        return f"file::{file_path}"
    def class_node_id(file_path, class_name):
        return f"class::{file_path}::{class_name}"
    def func_node_id(file_path, func_name, parent_class=None):
        if parent_class:
            return f"func::{file_path}::{parent_class}::{func_name}"
        return f"func::{file_path}::{func_name}"
    for chunk in chunks:
        if chunk['language'] != 'python':
            continue
        fp = chunk['file_path']
        if file_node_id(fp) not in file_nodes:
            file_nodes[file_node_id(fp)] = {
                'id': file_node_id(fp), 'type': 'File', 'file_path': fp
            }
            print(f"[KG] Added file node: {file_node_id(fp)}")
        if chunk['chunk_type'] == 'class':
            cname = chunk['function_name'] or chunk.get('class_name') or chunk.get('name') or chunk.get('parent_class') or 'UnknownClass'
            class_nodes[class_node_id(fp, cname)] = {
                'id': class_node_id(fp, cname), 'type': 'Class', 'name': cname, 'file_path': fp
            }
            print(f"[KG] Added class node: {class_node_id(fp, cname)}")
            kg_rels.append({'from': file_node_id(fp), 'to': class_node_id(fp, cname), 'type': 'CONTAINS'})
            print(f"[KG] Added CONTAINS rel: {file_node_id(fp)} -> {class_node_id(fp, cname)}")
            import re
            m = re.match(r'class\s+(\w+)\(([^)]*)\)', chunk['chunk_text'])
            if m:
                bases = [b.strip() for b in m.group(2).split(',') if b.strip()]
                for base in bases:
                    kg_rels.append({'from': class_node_id(fp, cname), 'to': f'class::{fp}::{base}', 'type': 'INHERITS'})
                    print(f"[KG] Added INHERITS rel: {class_node_id(fp, cname)} -> class::{fp}::{base}")
        elif chunk['chunk_type'] == 'function':
            fname = chunk['function_name']
            parent = chunk.get('parent_class')
            func_nodes[func_node_id(fp, fname, parent)] = {
                'id': func_node_id(fp, fname, parent), 'type': 'Function', 'name': fname, 'file_path': fp, 'parent_class': parent
            }
            print(f"[KG] Added function node: {func_node_id(fp, fname, parent)}")
            if parent:
                kg_rels.append({'from': class_node_id(fp, parent), 'to': func_node_id(fp, fname, parent), 'type': 'CONTAINS'})
                print(f"[KG] Added CONTAINS rel: {class_node_id(fp, parent)} -> {func_node_id(fp, fname, parent)}")
            else:
                kg_rels.append({'from': file_node_id(fp), 'to': func_node_id(fp, fname), 'type': 'CONTAINS'})
                print(f"[KG] Added CONTAINS rel: {file_node_id(fp)} -> {func_node_id(fp, fname)}")
            import re
            called = set(re.findall(r'(\w+)\s*\(', chunk['chunk_text']))
            for callee in called:
                if callee != fname:
                    kg_rels.append({'from': func_node_id(fp, fname, parent), 'to': func_node_id(fp, callee), 'type': 'CALLS'})
                    print(f"[KG] Added CALLS rel: {func_node_id(fp, fname, parent)} -> {func_node_id(fp, callee)}")
        elif chunk['chunk_type'] == 'import':
            import_name = chunk['chunk_text'].replace('import', '').replace('from', '').strip()
            import_node_id = f"import::{import_name}"
            kg_nodes.append({'id': import_node_id, 'type': 'Import', 'name': import_name})
            print(f"[KG] Added import node: {import_node_id}")
            kg_rels.append({'from': file_node_id(fp), 'to': import_node_id, 'type': 'IMPORTS'})
            print(f"[KG] Added IMPORTS rel: {file_node_id(fp)} -> {import_node_id}")
    kg_nodes.extend(file_nodes.values())
    kg_nodes.extend(class_nodes.values())
    kg_nodes.extend(func_nodes.values())
    print(f"[KG] Total nodes: {len(kg_nodes)}, Total rels: {len(kg_rels)}")

    # --- Neo4j Write ---
    try:
        print(f"[KG] Connecting to Neo4j at {NEO4J_URI}...")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        graph_label = repo["neo4j_graph_name"]
        with driver.session(database=NEO4J_DATABASE) as session:
            print(f"[KG] Deleting previous graph for label {graph_label}...")
            session.run(f"MATCH (n:{graph_label}) DETACH DELETE n")
            print(f"[KG] Creating nodes...")
            for node in kg_nodes:
                props = {k: v for k, v in node.items() if k != 'id' and k != 'type'}
                prop_str = ', '.join([f'{k}: ${k}' for k in props])
                session.run(f"CREATE (n:{graph_label}:{node['type']} {{id: $id, {prop_str}}})", {'id': node['id'], **props})
                print(f"[KG] Created node: {node['id']} [{node['type']}]")
            print(f"[KG] Creating relationships...")
            for rel in kg_rels:
                session.run(f"MATCH (a:{graph_label} {{id: $from}}), (b:{graph_label} {{id: $to}}) CREATE (a)-[:{rel['type']}]->(b)", {'from': rel['from'], 'to': rel['to']})
                print(f"[KG] Created rel: {rel['from']} -[{rel['type']}]-> {rel['to']}")
        driver.close()
        print(f"[KG] Knowledge graph for repo {repo['repo_name']} written to Neo4j as label {graph_label}")
    except Exception as e:
        print(f"[KG] Failed to write knowledge graph to Neo4j: {e}")
        neo4j_status = str(e)

    # --- Python KG Extraction ---
    # Only for Python files: extract nodes and relationships for Neo4j
    kg_nodes = []
    kg_rels = []
    file_nodes = {}
    class_nodes = {}
    func_nodes = {}
    # Helper: get or create node id
    def file_node_id(file_path):
        return f"file::{file_path}"
    def class_node_id(file_path, class_name):
        return f"class::{file_path}::{class_name}"
    def func_node_id(file_path, func_name, parent_class=None):
        if parent_class:
            return f"func::{file_path}::{parent_class}::{func_name}"
        return f"func::{file_path}::{func_name}"
    # Build nodes and relationships
    for chunk in chunks:
        if chunk['language'] != 'python':
            continue
        fp = chunk['file_path']
        if file_node_id(fp) not in file_nodes:
            file_nodes[file_node_id(fp)] = {
                'id': file_node_id(fp), 'type': 'File', 'file_path': fp
            }
        if chunk['chunk_type'] == 'class':
            cname = chunk['function_name'] or chunk.get('class_name') or chunk.get('name') or chunk.get('parent_class') or 'UnknownClass'
            class_nodes[class_node_id(fp, cname)] = {
                'id': class_node_id(fp, cname), 'type': 'Class', 'name': cname, 'file_path': fp
            }
            # CONTAINS: file -> class
            kg_rels.append({'from': file_node_id(fp), 'to': class_node_id(fp, cname), 'type': 'CONTAINS'})
            # INHERITS: class -> base(s)
            # Try to extract base class from chunk_text
            import re
            m = re.match(r'class\s+(\w+)\(([^)]*)\)', chunk['chunk_text'])
            if m:
                bases = [b.strip() for b in m.group(2).split(',') if b.strip()]
                for base in bases:
                    kg_rels.append({'from': class_node_id(fp, cname), 'to': f'class::{fp}::{base}', 'type': 'INHERITS'})
        elif chunk['chunk_type'] == 'function':
            fname = chunk['function_name']
            parent = chunk.get('parent_class')
            func_nodes[func_node_id(fp, fname, parent)] = {
                'id': func_node_id(fp, fname, parent), 'type': 'Function', 'name': fname, 'file_path': fp, 'parent_class': parent
            }
            # CONTAINS: file -> function, or class -> method
            if parent:
                kg_rels.append({'from': class_node_id(fp, parent), 'to': func_node_id(fp, fname, parent), 'type': 'CONTAINS'})
            else:
                kg_rels.append({'from': file_node_id(fp), 'to': func_node_id(fp, fname), 'type': 'CONTAINS'})
            # CALLS: function -> function (very basic, look for 'func(' in chunk_text)
            import re
            called = set(re.findall(r'(\w+)\s*\(', chunk['chunk_text']))
            for callee in called:
                if callee != fname:
                    kg_rels.append({'from': func_node_id(fp, fname, parent), 'to': func_node_id(fp, callee), 'type': 'CALLS'})
        elif chunk['chunk_type'] == 'import':
            # IMPORTS: file -> import
            import_name = chunk['chunk_text'].replace('import', '').replace('from', '').strip()
            import_node_id = f"import::{import_name}"
            kg_nodes.append({'id': import_node_id, 'type': 'Import', 'name': import_name})
            kg_rels.append({'from': file_node_id(fp), 'to': import_node_id, 'type': 'IMPORTS'})
    kg_nodes.extend(file_nodes.values())
    kg_nodes.extend(class_nodes.values())
    kg_nodes.extend(func_nodes.values())

    # --- Neo4j Write ---
    neo4j_status = "success"
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        graph_label = repo["neo4j_graph_name"]
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run(f"MATCH (n:{graph_label}) DETACH DELETE n")
            for node in kg_nodes:
                props = {k: v for k, v in node.items() if k != 'id' and k != 'type'}
                prop_str = ', '.join([f'{k}: ${k}' for k in props])
                session.run(f"CREATE (n:{graph_label}:{node['type']} {{id: $id, {prop_str}}})", {'id': node['id'], **props})
            for rel in kg_rels:
                session.run(f"MATCH (a:{graph_label} {{id: $from}}), (b:{graph_label} {{id: $to}}) CREATE (a)-[:{rel['type']}]->(b)", {'from': rel['from'], 'to': rel['to']})
        driver.close()
        print(f"Knowledge graph for repo {repo['repo_name']} written to Neo4j as label {graph_label}")
    except Exception as e:
        print(f"Failed to write knowledge graph to Neo4j: {e}")
        neo4j_status = str(e)

    # Load CodeT5-small model and tokenizer (once per request, can be optimized)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    model = AutoModel.from_pretrained("Salesforce/codet5-small").to(device)

    # Generate embeddings for each chunk
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get('chunk_text', '')
        if chunk_text:
            chunk['embedding'] = generate_codet5_embedding(chunk_text, tokenizer, model, device)
            if i < 3:  # Print first 3 embeddings for debug
                print(f"Embedding for chunk {i} (id={chunk['chunk_id']}): {chunk['embedding'][:5]}... (len={len(chunk['embedding'])})")
        else:
            chunk['embedding'] = None
            print(f"Chunk {i} (id={chunk['chunk_id']}) has empty text, skipping embedding.")

    # Store chunks in Postgres (without embedding for now, can add column if needed)
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        cur = conn.cursor()
        # Create table if not exists
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                chunk_id TEXT PRIMARY KEY,
                chunk_index INTEGER,
                file_path TEXT,
                chunk_type TEXT,
                language TEXT,
                start_line INTEGER,
                end_line INTEGER,
                parent_class TEXT,
                function_name TEXT,
                chunk_text TEXT
            )
        ''')
        # Clear previous entries for this repo (optional, for idempotency)
        cur.execute(f'DELETE FROM "{table_name}"')
        # Insert all chunks
        for chunk in chunks:
            cur.execute(f'''
                INSERT INTO "{table_name}" (
                    chunk_id, chunk_index, file_path, chunk_type, language, start_line, end_line, parent_class, function_name, chunk_text
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ''', (
                chunk['chunk_id'], chunk['chunk_index'], chunk['file_path'], chunk['chunk_type'], chunk['language'],
                chunk['start_line'], chunk['end_line'], chunk['parent_class'], chunk['function_name'], chunk['chunk_text']
            ))
        conn.commit()
        cur.close()
        conn.close()
        print(f"Stored {len(chunks)} chunks in Postgres table {table_name}")
    except Exception as e:
        print(f"Failed to store chunks in Postgres: {e}")
        return JSONResponse({"error": f"Failed to store chunks in Postgres: {e}"}, status_code=500)

    # Store embeddings in ChromaDB
    try:
        print("[EMBED] Connecting to ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection_name = repo["chroma_collection_name"]
        print(f"[EMBED] Storing embeddings in ChromaDB collection: {collection_name}")
        collections = chroma_client.list_collections()
        print(f"[EMBED] Existing collections: {[c.name for c in collections]}")
        if collection_name not in [c.name for c in collections]:
            print(f"[EMBED] Creating new collection: {collection_name}")
            collection = chroma_client.create_collection(collection_name)
        else:
            print(f"[EMBED] Using existing collection: {collection_name}")
            collection = chroma_client.get_collection(collection_name)
        # Remove all previous embeddings for this repo (idempotency)
        existing = collection.get()
        all_ids = existing.get('ids', [])
        print(f"[EMBED] Existing embeddings in collection: {len(all_ids)}")
        if all_ids:
            collection.delete(ids=all_ids)
            print(f"[EMBED] Deleted {len(all_ids)} old embeddings from collection.")
        # Add all chunk embeddings (only those with valid embeddings)
        valid_chunks = [chunk for chunk in chunks if chunk['embedding'] is not None]
        ids = [chunk['chunk_id'] for chunk in valid_chunks]
        embeddings = [chunk['embedding'] for chunk in valid_chunks]
        def safe_meta_value(val):
            return val if val is not None else ""
        metadatas = [
            {
                'chunk_index': chunk['chunk_index'],
                'file_path': safe_meta_value(chunk['file_path']),
                'chunk_type': safe_meta_value(chunk['chunk_type']),
                'language': safe_meta_value(chunk['language']),
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line'],
                'parent_class': safe_meta_value(chunk['parent_class']),
                'function_name': safe_meta_value(chunk['function_name'])
            }
            for chunk in valid_chunks
        ]
        documents = [chunk['chunk_text'] for chunk in valid_chunks]
        if ids and embeddings:
            print(f"[EMBED] Adding {len(ids)} embeddings to ChromaDB collection {collection_name}")
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
            print(f"[EMBED] Added {len(ids)} embeddings to ChromaDB collection {collection_name}")
        else:
            print("[EMBED] No valid embeddings to add to ChromaDB.")
    except Exception as e:
        print(f"[EMBED][ERROR] Failed to store embeddings in ChromaDB: {e}")
        return JSONResponse({"error": f"Failed to store embeddings in ChromaDB: {e}"}, status_code=500)

    print("Indexing and embedding complete.")
    result = {
        "success": True if neo4j_status == "success" else False,
        "repo_path": repo_path,
        "chunk_count": len(chunks),
        "error_files": error_files,
        "postgres_table": table_name,
        "chroma_collection": collection_name,
        "neo4j_status": neo4j_status
    }
    return result
# code_indexer/api.py
# FastAPI app to serve database connection status for the frontend

# Endpoint to list all registered repositories
@app.get("/api/repositories")
def get_repositories():
    init_control_plane()  # Ensure table exists
    rows = list_registered_repositories()
    # Map to dicts for JSON response
    columns = [
        "repo_id", "repo_name", "repo_path", "indexed_at",
        "postgres_table_name", "chroma_collection_name", "neo4j_graph_name"
    ]
    repos = [dict(zip(columns, row)) for row in rows]
    return {"repositories": repos}

# Helper to generate valid table/collection/graph names
def make_valid_name(base, maxlen=48):
    # Only allow alphanumeric and underscores, start with letter, lowercased
    name = re.sub(r'[^a-zA-Z0-9_]', '_', base)
    if not name or not name[0].isalpha():
        name = 'r_' + name
    return name[:maxlen].lower()

def sanitize_repo_name(repo_name):
    # Lowercase, replace non-alphanum with _, collapse multiple _
    name = re.sub(r'[^a-zA-Z0-9]', '_', repo_name.lower())
    name = re.sub(r'_+', '_', name).strip('_')
    if not name or not name[0].isalpha():
        name = 'r_' + name
    return name[:24]  # keep short for DB limits


# Endpoint to register a new repository
from fastapi.responses import JSONResponse
import hashlib
from datetime import datetime

@app.post("/api/register_repository")
async def register_repo(request: Request):
    data = await request.json()
    repo_name = data.get('repo_name')
    repo_path = data.get('repo_path')
    if not repo_name or not repo_path:
        return JSONResponse({"error": "repo_name and repo_path are required"}, status_code=400)
    # Generate unique repo_id (sha256 of path + timestamp)
    now = datetime.utcnow().isoformat()
    short_hash = hashlib.sha256((repo_path + now).encode()).hexdigest()[:6]
    repo_id = hashlib.sha256((repo_path + now).encode()).hexdigest()[:16]
    sanitized = sanitize_repo_name(repo_name)
    base = f"{sanitized}_{short_hash}"
    postgres_table = make_valid_name(base + "_metadata", maxlen=48)  # PostgreSQL max table name 63, use 48 for safety
    chroma_collection = make_valid_name(base + "_embeddings", maxlen=48)
    neo4j_graph = make_valid_name(base + "_graph", maxlen=48)
    try:
        register_repository(
            repo_id, repo_name, repo_path,
            postgres_table, chroma_collection, neo4j_graph
        )
        return {"success": True, "repo_id": repo_id,
                "postgres_table": postgres_table,
                "chroma_collection": chroma_collection,
                "neo4j_graph": neo4j_graph}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Allow local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/db_status")
def db_status():
    status = {}
    # PostgreSQL
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        conn.close()
        status['postgres'] = {"ok": True}
    except Exception as e:
        status['postgres'] = {"ok": False, "error": str(e)}
    # Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run('RETURN 1')
        status['neo4j'] = {"ok": True}
    except Exception as e:
        status['neo4j'] = {"ok": False, "error": str(e)}
    # ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        status['chroma'] = {"ok": True}
    except Exception as e:
        status['chroma'] = {"ok": False, "error": str(e)}
    # SQLite
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.close()
        status['sqlite'] = {"ok": True}
    except Exception as e:
        status['sqlite'] = {"ok": False, "error": str(e)}
    return JSONResponse(status)


# Serve the status page at '/'
@app.get("/", response_class=Response)
def serve_index():
    with open("web/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return Response(content=html, media_type="text/html")

@app.post("/api/clear_repositories")
def clear_repositories():
    try:
        clear_all_repositories()
        return {"success": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

from fastapi import Query
from starlette.responses import JSONResponse
import psycopg2

@app.get("/api/pg_table")
def get_pg_table(table: str = Query(...)):
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        cur = conn.cursor()
        cur.execute(f'SELECT * FROM "{table}" LIMIT 100')
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return {"success": True, "columns": columns, "rows": rows}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/pg_table.html", response_class=Response)
def serve_pg_table():
    with open("web/pg_table.html", "r", encoding="utf-8") as f:
        html = f.read()
    return Response(content=html, media_type="text/html")

@app.get("/api/chroma_collection")
def get_chroma_collection(collection: str = Query(...)):
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = [c.name for c in chroma_client.list_collections()]
        if collection not in collections:
            return {"success": False, "error": f"Collection {collection} does not exist. Please run indexing for this repository first."}
        col = chroma_client.get_collection(collection)
        # Get up to 100 entries
        results = col.get(limit=100)
        ids = results.get('ids', [])
        metadatas = results.get('metadatas', [])
        documents = results.get('documents', [])
        # Compose rows for table
        columns = ["chunk_id", "chunk_index", "file_path", "chunk_type", "language", "start_line", "end_line", "parent_class", "function_name", "chunk_text"]
        rows = []
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            row = [
                ids[i],
                meta.get('chunk_index'),
                meta.get('file_path'),
                meta.get('chunk_type'),
                meta.get('language'),
                meta.get('start_line'),
                meta.get('end_line'),
                meta.get('parent_class'),
                meta.get('function_name'),
                documents[i] if i < len(documents) else ''
            ]
            rows.append(row)
        return {"success": True, "columns": columns, "rows": rows}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/chroma_collection.html", response_class=Response)
def serve_chroma_collection():
    with open("web/chroma_collection.html", "r", encoding="utf-8") as f:
        html = f.read()
    return Response(content=html, media_type="text/html")

@app.post("/api/clear_psql_tables")
def clear_psql_tables():
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        cur = conn.cursor()
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        tables = [row[0] for row in cur.fetchall()]
        for table in tables:
            cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
        conn.commit()
        cur.close()
        conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/clear_chroma_collections")
def clear_chroma_collections():
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = chroma_client.list_collections()
        for col in collections:
            chroma_client.delete_collection(name=col.name)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/kg")
def get_kg(repo_id: str = Query(...)):
    # Find repo by id
    rows = list_registered_repositories()
    columns = [
        "repo_id", "repo_name", "repo_path", "indexed_at",
        "postgres_table_name", "chroma_collection_name", "neo4j_graph_name"
    ]
    repo = None
    for row in rows:
        r = dict(zip(columns, row))
        if r["repo_id"] == repo_id:
            repo = r
            break
    if not repo:
        return {"nodes": [], "edges": []}
    # Query Neo4j for this repo's graph
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        graph_label = repo["neo4j_graph_name"]
        with driver.session() as session:
            # Get all nodes
            node_result = session.run(f"MATCH (n:{graph_label}) RETURN n")
            nodes = []
            for rec in node_result:
                n = rec["n"]
                nodes.append({
                    "id": n["id"],
                    "label": n.get("name", n.get("id", n.labels[0])),
                    "group": list(n.labels)[-1] if n.labels else "Node"
                })
            # Get all relationships
            edge_result = session.run(f"MATCH (a:{graph_label})-[r]->(b:{graph_label}) RETURN a.id, type(r), b.id")
            edges = []
            for rec in edge_result:
                edges.append({
                    "from": rec["a.id"], "to": rec["b.id"], "label": rec["type(r)"]
                })
        driver.close()
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

@app.post("/api/clear_neo4j_graphs")
def clear_neo4j_graphs():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/kg_view.html", response_class=Response)
def serve_kg_view():
    with open("web/kg_view.html", "r", encoding="utf-8") as f:
        html = f.read()
    return Response(content=html, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("code_indexer.api:app", host="0.0.0.0", port=8000, reload=True)
