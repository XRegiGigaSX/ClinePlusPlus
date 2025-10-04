"""
Standalone file walker and chunker for Python, Java, Kotlin, and C++ using Tree-sitter.
- Walks a given directory recursively.
- Parses .py, .java, .kt, .cpp, .cc, .cxx, .hpp, .h files.
- Extracts functions, classes, imports, and module docstrings/comments as chunks.
- Prints chunk metadata for inspection.
"""


import os
import hashlib
import ast
import javalang
import re

EXT_LANGUAGE_MAP = {
    '.py': 'python',
    '.java': 'java',
    '.kt': 'kotlin',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.hpp': 'cpp',
    '.h': 'cpp',
}

def walk_source_files(root_dir):
    """Yield (file_path, language) for supported files under root_dir recursively."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            lang = EXT_LANGUAGE_MAP.get(ext)
            if lang:
                yield os.path.join(dirpath, fname), lang


def parse_file(file_path, language_name):
    """Parse a source file and yield code chunks with metadata, including chunk_id, chunk_index, parent_class, function_name."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        source_code = f.read()

    chunks = []
    chunk_index = 0

    def make_chunk_id(chunk_type, name, start_line, end_line):
        base = f"{file_path}|{chunk_type}|{name or ''}|{start_line}|{end_line}"
        return hashlib.sha256(base.encode()).hexdigest()

    if language_name == 'python':
        # Use ast for Python
        try:
            tree = ast.parse(source_code, filename=file_path)
        except Exception as e:
            return []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                start_line = getattr(node, 'lineno', 1)
                end_line = getattr(node, 'end_lineno', start_line)
                chunk = {
                    'chunk_id': make_chunk_id('import', None, start_line, end_line),
                    'file_path': file_path,
                    'chunk_index': chunk_index,
                    'chunk_text': source_code.splitlines()[start_line-1:end_line],
                    'chunk_type': 'import',
                    'language': language_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'parent_class': None,
                    'function_name': None
                }
                chunk['chunk_text'] = '\n'.join(chunk['chunk_text'])
                chunks.append(chunk)
                chunk_index += 1
            elif isinstance(node, ast.ClassDef):
                start_line = getattr(node, 'lineno', 1)
                end_line = getattr(node, 'end_lineno', start_line)
                chunk = {
                    'chunk_id': make_chunk_id('class', node.name, start_line, end_line),
                    'file_path': file_path,
                    'chunk_index': chunk_index,
                    'chunk_text': source_code.splitlines()[start_line-1:end_line],
                    'chunk_type': 'class',
                    'language': language_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'parent_class': None,
                    'function_name': None
                }
                chunk['chunk_text'] = '\n'.join(chunk['chunk_text'])
                chunks.append(chunk)
                chunk_index += 1
                # Methods
                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef):
                        sline = getattr(subnode, 'lineno', 1)
                        eline = getattr(subnode, 'end_lineno', sline)
                        subchunk = {
                            'chunk_id': make_chunk_id('function', subnode.name, sline, eline),
                            'file_path': file_path,
                            'chunk_index': chunk_index,
                            'chunk_text': '\n'.join(source_code.splitlines()[sline-1:eline]),
                            'chunk_type': 'function',
                            'language': language_name,
                            'start_line': sline,
                            'end_line': eline,
                            'parent_class': node.name,
                            'function_name': subnode.name
                        }
                        chunks.append(subchunk)
                        chunk_index += 1
            elif isinstance(node, ast.FunctionDef):
                start_line = getattr(node, 'lineno', 1)
                end_line = getattr(node, 'end_lineno', start_line)
                chunk = {
                    'chunk_id': make_chunk_id('function', node.name, start_line, end_line),
                    'file_path': file_path,
                    'chunk_index': chunk_index,
                    'chunk_text': '\n'.join(source_code.splitlines()[start_line-1:end_line]),
                    'chunk_type': 'function',
                    'language': language_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'parent_class': None,
                    'function_name': node.name
                }
                chunks.append(chunk)
                chunk_index += 1
        # Module docstring
        docstring = ast.get_docstring(tree)
        if docstring:
            chunk = {
                'chunk_id': make_chunk_id('docstring', None, 1, 1),
                'file_path': file_path,
                'chunk_index': chunk_index,
                'chunk_text': docstring,
                'chunk_type': 'docstring',
                'language': language_name,
                'start_line': 1,
                'end_line': 1,
                'parent_class': None,
                'function_name': None
            }
            chunks.append(chunk)
            chunk_index += 1
    elif language_name == 'java':
        # Use javalang for Java
        try:
            tree = javalang.parse.parse(source_code)
        except Exception as e:
            return []
        for path, node in tree.filter(javalang.tree.Import):
            start_line = getattr(node, 'position', (1,))[0] or 1
            chunk = {
                'chunk_id': make_chunk_id('import', None, start_line, start_line),
                'file_path': file_path,
                'chunk_index': chunk_index,
                'chunk_text': source_code.splitlines()[start_line-1] if start_line <= len(source_code.splitlines()) else '',
                'chunk_type': 'import',
                'language': language_name,
                'start_line': start_line,
                'end_line': start_line,
                'parent_class': None,
                'function_name': None
            }
            chunks.append(chunk)
            chunk_index += 1
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            start_line = getattr(node, 'position', (1,))[0] or 1
            chunk = {
                'chunk_id': make_chunk_id('class', node.name, start_line, start_line),
                'file_path': file_path,
                'chunk_index': chunk_index,
                'chunk_text': '',
                'chunk_type': 'class',
                'language': language_name,
                'start_line': start_line,
                'end_line': start_line,
                'parent_class': None,
                'function_name': None
            }
            chunks.append(chunk)
            chunk_index += 1
            # Methods
            for method in node.methods:
                mline = getattr(method, 'position', (start_line,))[0] or start_line
                subchunk = {
                    'chunk_id': make_chunk_id('function', method.name, mline, mline),
                    'file_path': file_path,
                    'chunk_index': chunk_index,
                    'chunk_text': '',
                    'chunk_type': 'function',
                    'language': language_name,
                    'start_line': mline,
                    'end_line': mline,
                    'parent_class': node.name,
                    'function_name': method.name
                }
                chunks.append(subchunk)
                chunk_index += 1
        # Java docstring: not supported by javalang, skip for now
    elif language_name in ('cpp', 'kotlin'):
        # Use regex for C++/Kotlin
        # C++: class, function, import (include)
        # Kotlin: class, function, import
        lines = source_code.splitlines()
        for i, line in enumerate(lines):
            # C++/Kotlin class
            m = re.match(r'\s*(class|struct)\s+(\w+)', line)
            if m:
                chunk = {
                    'chunk_id': make_chunk_id('class', m.group(2), i+1, i+1),
                    'file_path': file_path,
                    'chunk_index': chunk_index,
                    'chunk_text': line,
                    'chunk_type': 'class',
                    'language': language_name,
                    'start_line': i+1,
                    'end_line': i+1,
                    'parent_class': None,
                    'function_name': None
                }
                chunks.append(chunk)
                chunk_index += 1
            # C++ include
            if language_name == 'cpp' and line.strip().startswith('#include'):
                chunk = {
                    'chunk_id': make_chunk_id('import', None, i+1, i+1),
                    'file_path': file_path,
                    'chunk_index': chunk_index,
                    'chunk_text': line,
                    'chunk_type': 'import',
                    'language': language_name,
                    'start_line': i+1,
                    'end_line': i+1,
                    'parent_class': None,
                    'function_name': None
                }
                chunks.append(chunk)
                chunk_index += 1
            # Kotlin import
            if language_name == 'kotlin' and line.strip().startswith('import '):
                chunk = {
                    'chunk_id': make_chunk_id('import', None, i+1, i+1),
                    'file_path': file_path,
                    'chunk_index': chunk_index,
                    'chunk_text': line,
                    'chunk_type': 'import',
                    'language': language_name,
                    'start_line': i+1,
                    'end_line': i+1,
                    'parent_class': None,
                    'function_name': None
                }
                chunks.append(chunk)
                chunk_index += 1
            # C++/Kotlin function (very basic)
            m = re.match(r'.*\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{', line)
            if m:
                chunk = {
                    'chunk_id': make_chunk_id('function', m.group(1), i+1, i+1),
                    'file_path': file_path,
                    'chunk_index': chunk_index,
                    'chunk_text': line,
                    'chunk_type': 'function',
                    'language': language_name,
                    'start_line': i+1,
                    'end_line': i+1,
                    'parent_class': None,
                    'function_name': m.group(1)
                }
                chunks.append(chunk)
                chunk_index += 1
    return chunks


def chunk_repository(repo_path):
    """
    Walks and chunks all supported files in a repository.
    Returns a list of chunk metadata dicts and a list of error files.
    """
    all_chunks = []
    error_files = []
    for file_path, lang in walk_source_files(repo_path):
        try:
            chunks = parse_file(file_path, lang)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            error_files.append((file_path, str(e)))
    return all_chunks, error_files

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python code_chunker.py <path_to_repo>")
        sys.exit(1)
    repo_path = sys.argv[1]
    print(f"Walking and chunking supported files in: {repo_path}\n")
    chunks, error_files = chunk_repository(repo_path)
    for chunk in chunks:
        print(f"File: {chunk['file_path']} [{chunk['language']}]\n  chunk_id: {chunk['chunk_id']}\n  chunk_index: {chunk['chunk_index']}\n  chunk_type: {chunk['chunk_type']}\n  function_name: {chunk['function_name']}\n  parent_class: {chunk['parent_class']}\n  start_line: {chunk['start_line']}\n  end_line: {chunk['end_line']}\n  chunk_text: {chunk['chunk_text'][:60].replace(chr(10),' ')}{'...' if len(chunk['chunk_text'])>60 else ''}\n")
    print(f"\nTotal chunks created: {len(chunks)}")
    if error_files:
        print("\nErrors encountered in the following files:")
        for file_path, err in error_files:
            print(f"  {file_path}: {err}")

if __name__ == "__main__":
    main()
