import os
import pathlib

extensions = {'.py', '.cpp', '.c', '.h', '.hpp', '.cu', '.cuh', '.js', '.ts', '.rs', '.go', '.java'}

ignore_dirs = {
    'venv', '.venv', '__pycache__', 'build', 'dist', '.git', 
    'docs', 'logs', 'data', 'node_modules', '.vscode', '.idea'
}

total_lines = 0
files_count = 0

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    
    for file in files:
        ext = pathlib.Path(file).suffix
        if ext in extensions:
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    files_count += 1
            except Exception:
                pass

print(f"Total lines of code: {total_lines:,}")
print(f"Total source files: {files_count}")