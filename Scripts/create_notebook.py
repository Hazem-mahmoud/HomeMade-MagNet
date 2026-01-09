
import json
import os
import re
import ast

# Configuration
ROOT_SCRIPT = "main.py"
OUTPUT_NOTEBOOK = "MagNet_Run.ipynb"
SRC_DIR = "src"

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def resolve_module_path(module_name, current_file_path=None):
    """
    Resolves 'src.data.dataset' to '/path/to/src/data/dataset.py'.
    Also handles relative imports if current_file_path is provided.
    """
    script_dir = get_script_dir()
    
    parts = module_name.split('.')
    
    # Handle module aliases / submodules
    # Check if direct file exists: src/data/dataset.py
    candidate_path = os.path.join(script_dir, *parts) + ".py"
    if os.path.exists(candidate_path):
        return candidate_path
    
    # Check if it's a directory package: src/models -> src/models/__init__.py
    candidate_dir_init = os.path.join(script_dir, *parts, "__init__.py")
    if os.path.exists(candidate_dir_init):
        return candidate_dir_init
        
    return None

def find_local_imports(file_path):
    """
    Parses a python file and returns a list of local modules it imports.
    Returns: List of absolute file paths to dependencies.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    dependencies = []
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        print(f"Syntax error parsing {file_path}")
        return []

    base_dir = os.path.dirname(file_path)
    
    for node in ast.walk(tree):
        # Handle: from src.data import dataset
        # Handle: import src.models
        # Handle: from . import scaler_model (relative)
        
        module_path = None
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith('src'):
                    module_path = resolve_module_path(alias.name)
        
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith('src'):
                # from src.models import CNNNetwork
                # This could be importing a class from __init__.py or a submodule
                # We try to import the module itself
                module_path = resolve_module_path(node.module)
            
            elif node.level > 0:
                # Relative import: from . import cnn_model
                # Resolution is tricky without full context, simplify by assuming structure
                # . -> current dir, .. -> parent
                
                # Simple logic for this specific project structure:
                # calculated based on file_path location
                # Only handling simple '.' case for now as commonly seen in __init__.py
                if node.module:
                    # from .module import ...
                    # relative to current file's directory
                    target_name = node.module
                    candidate = os.path.join(base_dir, target_name + ".py")
                    if os.path.exists(candidate):
                        dependencies.append(candidate)
                        continue
                else:
                    # from . import module
                    # Usually found in __init__.py
                     for alias in node.names:
                        target_name = alias.name
                        candidate = os.path.join(base_dir, target_name + ".py")
                        if os.path.exists(candidate):
                            dependencies.append(candidate)
                            
        if module_path and module_path not in dependencies:
            dependencies.append(module_path)

    return dependencies

def build_dependency_graph(root_file):
    """
    BFS/DFS to find all dependencies. returns list [file1, file2, ...]
    """
    visited = set()
    stack = [root_file]
    ordered_files = [] # We want a topological sort essentially
    
    # We'll use a recursive helper for post-order traversal to ensure deps are valid
    
    def visit(current_file):
        if current_file in visited:
            return
        visited.add(current_file)
        
        deps = find_local_imports(current_file)
        for dep in deps:
            visit(dep)
        
        ordered_files.append(current_file)
    
    resolve_path = os.path.join(get_script_dir(), root_file)
    visit(resolve_path)
    
    return ordered_files

def comment_out_local_imports(content):
    """
    Comments out lines starting with 'from src' or 'import src' 
    or relative imports 'from .' 
    """
    lines = content.splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        # Regex for 'from src...' or 'import src...'
        if re.match(r'^(from\s+src\.|import\s+src\.|from\s+\.)', stripped):
            new_lines.append(f"# [NOTEBOOK_BUNDLER] {line}")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def main():
    script_dir = get_script_dir()
    main_script_path = os.path.join(script_dir, ROOT_SCRIPT)
    
    print(f"Building dependency graph for {ROOT_SCRIPT}...")
    all_files = build_dependency_graph(ROOT_SCRIPT)
    
    # Move main.py to the end explicitly just in case, though post-order should handle it
    # all_files includes imports AND main.py
    
    # The last element of ordered_files is the root (main.py) because of post-order
    # But let's verify
    if main_script_path in all_files:
        all_files.remove(main_script_path)
    
    files_to_process = all_files + [main_script_path]
    
    notebook_cells = []
    
    # 1. Header Cell
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# MagNet Standalone Notebook\n",
            "\n",
            "This notebook bundles the MagNet project `main.py` and its dependencies.\n",
            "It is designed to run in environments like Google Colab without requiring file uploads."
        ]
    })

    # 2. Setup Cell
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install dependencies\n",
            "!pip install -r requirements.txt\n",
            "import os\n",
            "import sys\n",
            "# Add current directory to path just in case\n",
            "sys.path.append(os.getcwd())"
        ]
    })
    
    # 2.5 Config Cell
    config_path = os.path.join(script_dir, "config", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        notebook_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Configuration\n",
                "Creating `config/config.yaml`..."
            ]
        })
        
        notebook_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "os.makedirs('config', exist_ok=True)\n",
                "config_content = \"\"\"" + config_content + "\"\"\"\n",
                "with open('config/config.yaml', 'w') as f:\n",
                "    f.write(config_content)"
            ]
        })
        print("Included config/config.yaml")
    
    # 3. Process each file
    for file_path in files_to_process:
        rel_path = os.path.relpath(file_path, script_dir)
        print(f"Processing {rel_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        modified_content = comment_out_local_imports(content)
        
        # Add Markdown header
        notebook_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### Module: `{rel_path}`\n",
                f"Content from local file: `{rel_path}`"
            ]
        })
        
        # Add Code cell
        notebook_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": modified_content.splitlines(keepends=True)
        })

    # 4. Construct Notebook JSON
    notebook_content = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    output_path = os.path.join(script_dir, OUTPUT_NOTEBOOK)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=1)
        
    print(f"Notebook generated at: {output_path}")

if __name__ == "__main__":
    main()
