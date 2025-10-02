"""
Analyze codebase to identify unused files and move them to UnUsedFiles folder
"""

import os
import ast
import re
from pathlib import Path
from typing import Set, Dict, List

class CodebaseAnalyzer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.used_files = set()
        self.all_python_files = set()
        self.imports_map = {}
        
    def find_all_python_files(self) -> Set[Path]:
        """Find all Python files in the codebase"""
        python_files = set()
        for file_path in self.root_path.rglob("*.py"):
            # Skip files already in UnUsedFiles
            if "UnUsedFiles" not in str(file_path):
                python_files.add(file_path)
        return python_files
    
    def extract_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find imports
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
                            for alias in node.names:
                                imports.add(f"{node.module}.{alias.name}")
            except SyntaxError:
                # Fallback to regex if AST parsing fails
                import_patterns = [
                    r'from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import',
                    r'import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)'
                ]
                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    imports.update(matches)
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return imports
    
    def is_file_referenced(self, file_path: Path, all_files: Set[Path]) -> bool:
        """Check if a file is referenced by other files"""
        file_stem = file_path.stem
        file_module = str(file_path.relative_to(self.root_path)).replace('/', '.').replace('.py', '')
        
        for other_file in all_files:
            if other_file == file_path:
                continue
                
            imports = self.extract_imports(other_file)
            
            # Check various import patterns
            for imp in imports:
                if (file_stem in imp or 
                    file_module in imp or 
                    file_path.name.replace('.py', '') in imp):
                    return True
        
        return False
    
    def analyze_codebase(self) -> Dict[str, List[Path]]:
        """Analyze the entire codebase to identify used and unused files"""
        all_files = self.find_all_python_files()
        
        # Core files that should never be moved
        core_files = {
            'main.py', 'production_main.py', 'start_server.py', 
            '__init__.py', 'config.py', 'shared_models.py'
        }
        
        # Files that are entry points or actively used
        active_files = set()
        unused_files = set()
        
        print(f"Analyzing {len(all_files)} Python files...")
        
        for file_path in all_files:
            file_name = file_path.name
            
            # Always keep core files
            if file_name in core_files:
                active_files.add(file_path)
                continue
            
            # Check if file is referenced by others
            if self.is_file_referenced(file_path, all_files):
                active_files.add(file_path)
            else:
                # Additional checks for specific patterns
                if self.is_likely_active_file(file_path):
                    active_files.add(file_path)
                else:
                    unused_files.add(file_path)
        
        return {
            'active': sorted(list(active_files)),
            'unused': sorted(list(unused_files))
        }
    
    def is_likely_active_file(self, file_path: Path) -> bool:
        """Check if file is likely active based on content and patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Files with FastAPI routes
            if 'FastAPI' in content or '@app.' in content or '@router.' in content:
                return True
            
            # Files with main execution
            if '__name__ == "__main__"' in content:
                return True
            
            # Recently created evaluation files
            if any(keyword in file_path.name for keyword in [
                'real_model_evaluator', 'comprehensive_analysis_generator', 
                'fix_analysis_structure', 'sample_queries_data'
            ]):
                return True
            
            # Database and model files
            if any(keyword in str(file_path) for keyword in [
                'database', 'models', 'api'
            ]):
                return True
                
        except Exception:
            pass
        
        return False
    
    def move_unused_files(self, unused_files: List[Path]):
        """Move unused files to UnUsedFiles folder"""
        unused_dir = self.root_path / "UnUsedFiles"
        unused_dir.mkdir(exist_ok=True)
        
        moved_files = []
        
        for file_path in unused_files:
            try:
                # Create relative path structure in UnUsedFiles
                relative_path = file_path.relative_to(self.root_path)
                target_path = unused_dir / relative_path
                
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move the file
                file_path.rename(target_path)
                moved_files.append((str(file_path), str(target_path)))
                print(f"Moved: {file_path} -> {target_path}")
                
            except Exception as e:
                print(f"Error moving {file_path}: {e}")
        
        return moved_files

def main():
    """Main function to analyze and clean up codebase"""
    root_path = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code"
    
    analyzer = CodebaseAnalyzer(root_path)
    results = analyzer.analyze_codebase()
    
    print("\n" + "="*60)
    print("CODEBASE ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nACTIVE FILES ({len(results['active'])}):")
    for file_path in results['active']:
        print(f"  ✓ {file_path.relative_to(Path(root_path))}")
    
    print(f"\nUNUSED FILES ({len(results['unused'])}):")
    for file_path in results['unused']:
        print(f"  ✗ {file_path.relative_to(Path(root_path))}")
    
    if results['unused']:
        print(f"\nMoving {len(results['unused'])} unused files to UnUsedFiles folder...")
        moved_files = analyzer.move_unused_files(results['unused'])
        print(f"Successfully moved {len(moved_files)} files")
    else:
        print("\nNo unused files found!")
    
    print("\nCodebase cleanup completed!")

if __name__ == "__main__":
    main()
