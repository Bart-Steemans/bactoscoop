"""Validate an installed distribution and execute copied walkthroughs.

Run with the clean environment's interpreter. This script deliberately does not
add the source checkout to sys.path. Scientific outputs live under --output.
"""
import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['BACTOSCOOP_LOG_LEVEL'] = 'WARNING'
os.environ.pop('PYTHONPATH', None)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tests', action='store_true')
    parser.add_argument('--tests-only', action='store_true')
    args = parser.parse_args()
    examples = args.examples.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import bactoscoop
    installed = Path(bactoscoop.__file__).resolve()
    assert Path(os.path.normcase(installed)).is_relative_to(Path(os.path.normcase(Path(sys.prefix).resolve()))), f'Not the clean installed package: {installed}'
    result = {'python':sys.version, 'interpreter':sys.executable, 'version':version('bactoscoop'),
              'import_path':str(installed), 'installed_distribution_verified':True,
              'segmentation':'Supplied masks reused; fresh Omnipose inference is not performed.',
              'tests':{}, 'walkthroughs':{}}
    report = output/'results.json'
    def save():
        report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    save()
    failed = False
    if args.tests or args.tests_only:
        tests_dir = examples.parent/'tests'
        namespace = types.ModuleType('tests')
        namespace.__path__ = [str(tests_dir)]
        sys.modules['tests'] = namespace
        for test_file in sorted(tests_dir.glob('test*.py')):
            filename = test_file.name
            spec = importlib.util.spec_from_file_location('tests.'+Path(filename).stem,tests_dir/filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with (output/(filename+'.log')).open('w',encoding='utf-8') as log:
                completed = unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
            result['tests'][filename] = {'run':completed.testsRun, 'failures':len(completed.failures),
                                        'errors':len(completed.errors), 'skipped':len(completed.skipped)}
            failed |= not completed.wasSuccessful()
            save()
    if not args.tests_only:
        import nbformat
        from nbclient import NotebookClient
        for source in sorted(examples.glob('*_walkthrough.ipynb')):
            started = time.monotonic()
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            notebook = nbformat.read(source,as_version=4)
            code = '''import sys, os
from pathlib import Path
from importlib.metadata import version
import bactoscoop
assert Path(os.path.normcase(Path(bactoscoop.__file__).resolve())).is_relative_to(Path(os.path.normcase(Path(sys.prefix).resolve())))
print("Installed release:", version("bactoscoop"), bactoscoop.__file__)
'''
            notebook.cells.insert(0,nbformat.v4.new_code_cell(code))
            notebook.cells.append(nbformat.v4.new_code_cell('''import json
validated_collection = globals().get("ic", globals().get("collection"))
assert validated_collection is not None
(RUN_DIR / "release-processing-summary.json").write_text(
    json.dumps(validated_collection.processing_summary(), indent=2), encoding="utf-8")
'''))
            for cell in notebook.cells:
                if cell.cell_type=='code':
                    cell.source = cell.source.replace('Path.home() / "bactoscoop_runs"',repr(str(output/'runs')))
                    # Keep the Path constructor: use a validation destination instead of the user's home.
                    cell.source = cell.source.replace(repr(str(output/'runs')),f'Path({str(output/"runs")!r})')
            kernel_env = dict(os.environ, BACTOSCOOP_EXAMPLES_DIR=str(examples))
            client = NotebookClient(notebook,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(output)}},allow_errors=False)
            error = None
            try:
                client.execute(env=kernel_env)
            except Exception as exc:
                error = str(exc)
                failed = True
            executed = output/source.name
            nbformat.write(notebook,executed)
            summary = {'seconds':round(time.monotonic()-started,2),'error':error,
                       'source_sha256':original_hash, 'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==original_hash,
                       'executed_code_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells)}
            if error is None:
                # Read the actual exported table and processing summary from the isolated run.
                import pandas as pd
                dataset = source.stem.replace('_walkthrough','')
                runs = sorted((output/'runs').rglob(dataset))
                features = list(runs[-1].glob('*features.parquet')) if runs else []
                if not features:
                    raise AssertionError(f'No saved feature table for {source.name}')
                table = pd.read_parquet(features[0])
                assert len(table)>0, 'Empty walkthrough output'
                summary.update(rows=len(table),columns=len(table.columns),output_directory=str(runs[-1]))
                processing = json.loads((runs[-1]/'release-processing-summary.json').read_text(encoding='utf-8'))
                summary['processing_summary'] = processing
            result['walkthroughs'][source.name] = summary
            save()
            print(source.name,summary,flush=True)
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(1 if failed else 0)

if __name__=='__main__':
    main()
