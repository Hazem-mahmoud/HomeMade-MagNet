import json
import os
import re
import ast

# Configuration
ROOT_SCRIPT = "main.py"
OUTPUT_NOTEBOOK = "MagNet_Run_Colab.ipynb"
SRC_DIR = "src"

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def resolve_module_path(module_name, current_file_path=None):
    script_dir = get_script_dir()
    parts = module_name.split('.')
    
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
        module_path = None
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith('src'):
                    module_path = resolve_module_path(alias.name)
        
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith('src'):
                pkg_path = resolve_module_path(node.module)
                if pkg_path and pkg_path not in dependencies:
                     dependencies.append(pkg_path)
                
                for alias in node.names:
                     potential_module_name = f"{node.module}.{alias.name}"
                     submodule_path = resolve_module_path(potential_module_name)
                     if submodule_path and submodule_path not in dependencies:
                         dependencies.append(submodule_path)
            
            elif node.level > 0:
                if node.module:
                     target_name = node.module
                     candidate = os.path.join(base_dir, target_name + ".py")
                     if os.path.exists(candidate):
                        dependencies.append(candidate)
                        continue
                else:
                     for alias in node.names:
                        target_name = alias.name
                        candidate = os.path.join(base_dir, target_name + ".py")
                        if os.path.exists(candidate):
                            dependencies.append(candidate)
                            
        if module_path and module_path not in dependencies:
            dependencies.append(module_path)

    return dependencies

def build_dependency_graph(root_file):
    visited = set()
    ordered_files = [] 
    
    def visit(current_file):
        if current_file in visited:
            return
        visited.add(current_file)
        print(f"DEBUG: Visiting {current_file}")
        
        deps = find_local_imports(current_file)
        for dep in deps:
            visit(dep)
        
        ordered_files.append(current_file)
    
    resolve_path = os.path.join(get_script_dir(), root_file)
    visit(resolve_path)
    return ordered_files

def comment_out_local_imports(content):
    lines = content.splitlines()
    new_lines = []
    flattened_modules = []
    skip_block = False
    block_indent = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Check if we are currently skipping a block
        if skip_block:
            # Check indentation to see if we are still in the block
            # If line is empty, it's still part of the block (or logic doesn't matter)
            if not stripped:
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= block_indent:
                # We dropped back to outer level, stop skipping
                skip_block = False
            else:
                # Still inside the block, skip this line
                continue

        # Check for __name__ == "__main__" start
        if line.strip() == 'if __name__ == "__main__":' or line.strip() == "if __name__ == '__main__':":
             skip_block = True
             block_indent = len(line) - len(line.lstrip())
             new_lines.append(f"# [NOTEBOOK_BUNDLER] Removed __main__ block execution")
             continue

        # Standard processing
        match_from_src = re.match(r'^from\s+(src\.[a-zA-Z0-9_\.]+)\s+import\s+(.+)', stripped)

        if re.match(r'^(from\s+src\.|import\s+src\.|from\s+\.)', stripped):
            # Comment out local import
            new_lines.append(f"# [NOTEBOOK_BUNDLER] {line}")
            
            if match_from_src:
                attributes = match_from_src.group(2)
                for attr in attributes.split(','):
                    attr = attr.strip()
                    if attr[0].islower(): # Heuristic: modules are lowercase
                         flattened_modules.append(attr)
                         
        elif '__file__' in line:
             # Comment out lines using __file__
             new_lines.append(f"# [NOTEBOOK_BUNDLER] (Ref to __file__ removed) {line}")
             if '=' in line:
                 # Capture indentation
                 indent = line[:len(line) - len(line.lstrip())]
                 var_name = line.split('=')[0].strip()
                 new_lines.append(f"{indent}{var_name} = 'notebook_mode'")
        else:
            new_lines.append(line)
            
    return "\n".join(new_lines), flattened_modules

def apply_namespace_reduction(content, modules_to_flatten):
    """
    Replaces 'loader.func()' with 'func()' for modules in the list.
    """
    for mod in modules_to_flatten:
        pattern = r'\b' + re.escape(mod) + r'\.'
        content = re.sub(pattern, '', content)
    return content

def apply_colab_patches(filename, content):
    """
    Applies strict replacements to match the manual fixes made in the copy notebook.
    """
    import textwrap
    basename = os.path.basename(filename)

    if basename == "dataset.py":
        print("    -> Applying Colab patches to dataset.py")
        content = re.sub(
            r'try:.*?from\s+\.\s+import\s+loader.*?except ImportError:.*?import\s+preprocessing', 
            '', 
            content, 
            flags=re.DOTALL
        )
        content = re.sub(
            r'config_path\s*=\s*os\.path\.abspath\(.*\)',
            "config_path = os.path.abspath( '/content/config/config.yaml')",
            content
        )

    elif basename == "main.py":
        print("    -> Applying Colab patches to main.py")
        
        # 1. Update argparse default
        content = content.replace(
            "default='Scripts/config/config.yaml'", 
            "default='/content/config/config.yaml'"
        )
        
        # 2. Patch main definition for notebook execution
        content = content.replace("def main():", "def main(args=None):")
        content = re.sub(r'(\w+)\.parse_args\(\)', r'\1.parse_args(args)', content)
        
        # 3. Comment out __main__ execution (redundant now with generic remover, but safe to keep specific check)
        content = content.replace('if __name__ == "__main__":', '# if __name__ == "__main__":')
        content = content.replace('    main()', '#    main()')

        # 4. FIX LOOP SCOPE: Use run_config copy
        if "train_config['save_dir'] = os.path.join(train_config['save_dir'], model_name)" in content:
            content = content.replace(
                "train_config['save_dir'] = os.path.join(train_config['save_dir'], model_name)",
                "current_save_dir = os.path.join(train_config['save_dir'], model_name)"
            )
            content = content.replace(
                "if not os.path.exists(train_config['save_dir']):",
                "if not os.path.exists(current_save_dir):"
            )
            content = content.replace(
                "os.makedirs(train_config['save_dir'])",
                "os.makedirs(current_save_dir)"
            )
            
            # Use textwrap.dedent to clean the string, then indent it with 8 spaces
            replacement_code = textwrap.dedent("""
                # Update config just for this run (be careful not to mutate original permanently if looping)
                run_config = train_config.copy()
                run_config['save_dir'] = current_save_dir
                
                model = model.to(device)
                trained_model, history = train_model(model, train_loader, val_loader, run_config, device)
            """)
            # Indent with 8 spaces (assuming inside main loop)
            replacement_code = textwrap.indent(replacement_code, '        ')
            
            content = re.sub(
                r'model\s*=\s*model\.to\(device\)\s*trained_model,\s*history\s*=\s*train_model\(.*?\)',
                replacement_code,
                content,
                flags=re.DOTALL
            )

        # 5. FIX INDENTATION / LOGIC for CNN/Transformer
        # We need to replace the indented comment blocks with properly indented 'elif' blocks (8 spaces)
        # CRITICAL: Do NOT include 'model = ...' in replacement as we want to keep the logic in main.py 
        # (which has the correct stats passing). We only fix the Header.
        
        # CNN Block Header
        cnn_block = textwrap.dedent("""
            elif model_name == 'cnn':
                # Input: B (1) -> Output: Log Loss (1) (Scalar)
                model_conf = config['models']['cnn']
        """)
        # Indent: elif needs 8 spaces. Body needs 12.
        # dedent makes 'elif' 0. indent(8) makes it 8.
        # The body lines inside the string have 4 spaces relative to elif.
        # So indent(8) makes them 12. Perfect.
        cnn_block = textwrap.indent(cnn_block, '        ')
        
        # Search for the specific pattern in main.py identifying the CNN block
        # Use [ \t]* to capture indentation but NOT newlines from previous lines
        cnn_pattern = r'([ \t]*)# Input: B \(1\)\s*\n\s*# Output: Log Loss \(1\) \(Scalar\)\s*\n\s*model_conf = config\[\'models\'\]\[\'cnn\'\]'
        
        if re.search(cnn_pattern, content):
            print("      -> Fixing CNN block indentation (Header only)")
            # Use lstrip('\n') to remove the first empty line from """...""", but keep indentation.
            # rstrip() to remove trailing whitespace/newlines is usually fine and safe.
            content = re.sub(cnn_pattern, cnn_block.lstrip('\n').rstrip(), content, count=1)
        
        # Transformer Block Header
        transformer_block = textwrap.dedent("""
            elif model_name == 'transformer':
                # Input: B (1) -> Output: Log Loss (1) (Scalar)
                model_conf = config['models']['transformer']
        """)
        transformer_block = textwrap.indent(transformer_block, '        ')
        
        transformer_pattern = r'([ \t]*)# Input: B \(1\)\s*\n\s*# Output: Log Loss \(1\) \(Scalar\)\s*\n\s*model_conf = config\[\'models\'\]\[\'transformer\'\]'
        
        if re.search(transformer_pattern, content):
            print("      -> Fixing Transformer block indentation (Header only)")
            content = re.sub(transformer_pattern, transformer_block.lstrip('\n').rstrip(), content, count=1)

    return content

def main():
    script_dir = get_script_dir()
    main_script_path = os.path.join(script_dir, ROOT_SCRIPT)
    
    print(f"Building dependency graph for {ROOT_SCRIPT}...")
    all_files = build_dependency_graph(ROOT_SCRIPT)
    
    if main_script_path in all_files:
        all_files.remove(main_script_path)
        
    # FORCE INCLUDE MODELS & UTILS
    models_dir = os.path.join(script_dir, "src", "models")
    extra_files = [
        os.path.join(models_dir, "cnn_model.py"),
        os.path.join(models_dir, "transformer_model.py"),
        os.path.join(models_dir, "sequence_model.py"),
        os.path.join(models_dir, "seq2seq_model.py"),
        os.path.join(models_dir, "scaler_model.py"),
        os.path.join(script_dir, "src", "data", "dataset.py"),
        os.path.join(script_dir, "src", "training", "train.py"),
        os.path.join(script_dir, "src", "training", "evaluate.py"),
        os.path.join(script_dir, "src", "utils", "visualization.py")
    ]
    for m in extra_files:
        if m not in all_files and os.path.exists(m):
            print(f"Manually adding {os.path.basename(m)}")
            all_files.append(m)
    
    # Process main.py last
    files_to_process = all_files + [main_script_path]
    
    notebook_cells = []
    
    # 1. Header Cell
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "header_md"},
        "source": [
            "# MagNet Standalone Notebook\n",
            "\n",
            "This notebook bundles the MagNet project `main.py` and its dependencies.\n",
            "It is designed to run in environments like Google Colab without requiring file uploads."
        ]
    })

    # 2. Drive Mount Cell
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "drive_mount"},
        "outputs": [],
        "source": [
            "from google.colab import drive\n",
            "drive.mount('/content/drive')"
        ]
    })

    # 3. Dependencies
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "setup_env"},
        "outputs": [],
        "source": [
            "# Install dependencies\n",
            "#!pip install -r requirements.txt\n",
            "import os\n",
            "import sys\n",
            "# Add current directory to path just in case\n",
            "sys.path.append(os.getcwd())"
        ]
    })
    
    # 4. Config Cell (Patched paths)
    config_path = os.path.join(script_dir, "config", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Patch Config Content for Colab
        config_content = config_content.replace('data/raw', '/content/config/raw')
        config_content = config_content.replace('data/processed', '/content/config/processed')
        
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
            "metadata": {"id": "config_create"},
            "outputs": [],
            "source": [
                "import os\n",
                "os.makedirs('config', exist_ok=True)\n",
                "config_content = \"\"\"" + config_content + "\"\"\"\n",
                "with open('config/config.yaml', 'w') as f:\n",
                "    f.write(config_content)"
            ]
        })
    
    # 5. Process Files
    for file_path in files_to_process:
        rel_path = os.path.relpath(file_path, script_dir)
        print(f"Processing {rel_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        modified_content, flattened_modules = comment_out_local_imports(content)
        
        # Force add commonly used modules that are now global in the notebook
        flattened_modules.extend(['loader', 'preprocessing'])
        flattened_modules = list(set(flattened_modules))
        
        if flattened_modules:
            modified_content = apply_namespace_reduction(modified_content, flattened_modules)
        
        # Apply strict manual patches
        modified_content = apply_colab_patches(file_path, modified_content)
        
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
            "metadata": {"id": f"module_{os.path.basename(file_path).replace('.','_')}"},
            "outputs": [],
            "source": modified_content.splitlines(keepends=True)
        })

    # 6. Example Execution Cell
    notebook_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Run Training\n",
            "Call the `main()` function with arguments as a list of strings."
        ]
    })
    
    notebook_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "run_example"},
        "outputs": [],
        "source": [
            "# Example: Train CNN Model\n",
            "# Make sure the data file path is correct\n",
            "data_file = \"/content/drive/MyDrive/Colab Notebooks/3C90_TX-25-15-10_Data1_Cycle.mat\"\n",
            "if not os.path.exists(data_file):\n",
            "    print(f\"Warning: {data_file} not found. Please upload it or fix the path.\")\n",
            "\n",
            "# Arguments: --data <path> --model <model_name> --epochs <N>\n",
            "args = ['--data', data_file, '--model', 'cnn', '--epochs', '10']\n",
            "\n",
            "try:\n",
            "    main(args)\n",
            "except SystemExit:\n",
            "    # argparse raises SystemExit on help or error, catch it so notebook doesn't crash\n",
            "    pass"
        ]
    })

    # 7. Dump JSON
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
            },
            "colab": {
                "provenance": []
            },
             "accelerator": "T4"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    output_path = os.path.join(script_dir, OUTPUT_NOTEBOOK)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2)
        
    print(f"Notebook generated at: {output_path}")

if __name__ == "__main__":
    main()
