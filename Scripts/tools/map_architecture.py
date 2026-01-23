import ast
import os
import sys

def get_type_hint_name(node):
    """Helper to extract type hint string from AST node."""
    if node is None:
        return "Any"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        return f"{get_type_hint_name(node.value)}[{get_type_hint_name(node.slice)}]"
    if isinstance(node, ast.Attribute):
        return f"{get_type_hint_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return str(node.value)
    return "ComplexType"

def scan_file(file_path, src_root):
    """Scans a single file for imports and function signatures."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return None

    # Rel path for display
    rel_path = os.path.relpath(file_path, src_root).replace("\\", "/")
    module_name = rel_path.replace("/", ".").replace(".py", "")
    
    analysis = {
        "module": module_name,
        "imports": [],
        "functions": [],
        "classes": []
    }

    for node in ast.walk(tree):
        # Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                analysis["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                analysis["imports"].append(node.module)

        # Functions (Top-level)
        if isinstance(node, ast.FunctionDef):
            # args
            args_list = []
            for arg in node.args.args:
                input_name = arg.arg
                input_type = get_type_hint_name(arg.annotation)
                args_list.append(f"{input_name}: {input_type}")
            
            # return type
            ret_type = get_type_hint_name(node.returns)
            
            analysis["functions"].append({
                "name": node.name,
                "args": args_list,
                "return": ret_type
            })

    return analysis

def generate_mermaid(analyses):
    """Generates Mermaid Class Diagram."""
    lines = ["classDiagram"]
    
    # Create classes for modules
    for item in analyses:
        mod_clean = item["module"].replace(".", "_")
        lines.append(f"class {mod_clean} {{")
        lines.append(f"    <<{item['module']}>>")
        for func in item["functions"]:
            args_str = ", ".join(func["args"])
            # Escape chars if needed, but simple for now
            # Mermaid methods: name(args) return
            # We'll truncate args if too long for display? No, Architect wants details.
            lines.append(f"    +{func['name']}({args_str}) {func['return']}")
        lines.append("}")
    
    # Create relationships (Dependencies)
    # We only care about internal deps (src.*) roughly, or external if significant.
    # To keep map clean, let's filter imports to only those in the list of modules we scanned.
    
    known_modules = {item["module"] for item in analyses}
    
    for item in analyses:
        mod_clean = item["module"].replace(".", "_")
        for imp in item["imports"]:
            # Check if import is one of our internal modules
            # Handle 'src.data.loader' matching 'src.data.loader'
            # Also handle relative imports resolved to absolute?
            # AST ImportFrom gives 'module'.
            
            # Simple matching
            matched = None
            for km in known_modules:
                if imp == km or imp.endswith(km) or km.endswith(imp):
                    # Avoid self-ref
                    if km != item["module"]:
                        matched = km
                        break
            
            if matched:
                match_clean = matched.replace(".", "_")
                lines.append(f"{mod_clean} ..> {match_clean} : imports")
                
    return "\n".join(lines)

def main():
    # Assume script is in Scripts/tools/
    # We want to scan Scripts/src/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..")) # Scripts/
    src_dir = os.path.join(root_dir, "src")
    docs_dir = os.path.join(root_dir, "docs")
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        
    out_file = os.path.join(docs_dir, "system_map.mmd")
    
    all_analyses = []
    
    # Walk src dir
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                full_path = os.path.join(root, file)
                analysis = scan_file(full_path, src_dir) # Use src_dir as root for naming
                if analysis:
                    # Prefix module with 'src' to match typical import usage if we scanned from src
                    # scan_file does relpath. If file is src/data/loader.py, relpath is data/loader.py -> data.loader
                    # Let's prepend 'src.'
                    analysis["module"] = "src." + analysis["module"]
                    all_analyses.append(analysis)
                    
    print(f"Scanned {len(all_analyses)} files.")
    
    mermaid_content = generate_mermaid(all_analyses)
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(mermaid_content)
        
    print(f"Map generated at: {out_file}")

if __name__ == "__main__":
    main()
