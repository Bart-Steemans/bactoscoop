"""Read-only repository inspection; save evidence for the documentation build."""
import ast
import hashlib
import json
from pathlib import Path
import openpyxl

ROOT = Path(__file__).resolve().parent
from docs_config import PACKAGE
REPORT = ROOT / 'reports'
REPORT.mkdir(exist_ok=True)

inventory = []
api = []
features = []
for path in sorted(PACKAGE.glob('bactoscoop/*.py')):
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    inventory.append({'file': str(path.relative_to(PACKAGE)), 'lines': len(source.splitlines()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            api.append({'module': path.stem, 'class': None, 'name': node.name, 'args': ast.unparse(node.args), 'line': node.lineno, 'end': node.end_lineno, 'doc': ast.get_docstring(node) or ''})
        elif isinstance(node, ast.ClassDef):
            api.append({'module': path.stem, 'class': None, 'name': node.name, 'kind': 'class', 'line': node.lineno, 'end': node.end_lineno, 'doc': ast.get_docstring(node) or ''})
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    api.append({'module': path.stem, 'class': node.name, 'name': method.name, 'args': ast.unparse(method.args), 'line': method.lineno, 'end': method.end_lineno, 'doc': ast.get_docstring(method) or ''})
                    if path.stem == 'features':
                        for assignment in ast.walk(method):
                            if not isinstance(assignment, ast.Assign):
                                continue
                            for target in (child for item in assignment.targets for child in ast.walk(item)):
                                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute) and target.value.attr.endswith('_features') and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                                    parent = next((p for p in ast.walk(method) if isinstance(p, ast.If) and assignment in list(ast.walk(p)) and 'all_data' in ast.unparse(p.test)), None)
                                    features.append({'category': method.name, 'key': target.slice.value, 'line': assignment.lineno, 'expression': ast.unparse(assignment.value), 'all_data_only': parent is not None})

workbook = openpyxl.load_workbook(PACKAGE / 'SingleCellFeatures.xlsx', data_only=True, read_only=True)
spreadsheet = {sheet.title: [[str(value) if value is not None else '' for value in row] for row in sheet.iter_rows(values_only=True)] for sheet in workbook}
notebooks = []
for path in PACKAGE.glob('examples/*.ipynb'):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    figures = sum('image/png' in output.get('data', {}) for cell in notebook['cells'] for output in cell.get('outputs', []))
    notebooks.append({'file': str(path.relative_to(PACKAGE)), 'cells': len(notebook['cells']), 'saved_pngs': figures, 'markdown': [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'markdown'], 'code': [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']})
for pattern in ['*.md','*.toml','requirements.txt','LICENSE','tests/*.py','examples/*.py','paralell*.py']:
    for path in sorted(PACKAGE.glob(pattern)):
        text = path.read_text(encoding='utf-8-sig')
        inventory.append({'file': str(path.relative_to(PACKAGE)), 'lines': len(text.splitlines()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
data = {'package': str(PACKAGE), 'inventory': inventory, 'api': api, 'features': features, 'spreadsheet': spreadsheet, 'notebooks': notebooks}
(REPORT / 'inspection.json').write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'files': len(inventory), 'api_entries': len(api), 'feature_assignments': len(features), 'feature_categories': {k:sum(f['category']==k for f in features) for k in sorted({f['category'] for f in features})}, 'spreadsheet': {k:len(v) for k,v in spreadsheet.items()}, 'notebooks':[{k:v for k,v in n.items() if k not in ['markdown','code']} for n in notebooks]}, indent=2))
