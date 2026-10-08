"""Rebuild documentation from guides and current local package sources."""
import subprocess
import sys
import shutil
import argparse
from pathlib import Path
from docs_config import PACKAGE

ROOT = Path(__file__).resolve().parent
def run(*args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true', help='Refresh repository docs/ and the publication ZIP.')
    parser.add_argument('--serve', action='store_true', help='Open the rebuilt documentation locally.')
    args = parser.parse_args()
    run('inspect_package.py')
    run('write_guides.py')
    run('generate_reference.py')
    # Fresh output prevents removed pages and old notebook figures surviving a rebuild.
    output = (ROOT/'build/html').resolve()
    assert output.is_relative_to(ROOT.resolve()) and output.name == 'html' and output.parent.name == 'build'
    if output.exists():
        shutil.rmtree(output)
    run('-m','sphinx','-b','html','-W','--keep-going','-E','-a','-d',str(ROOT/'build/doctrees'),
        str(ROOT/'source'),str(ROOT/'build/html'))
    # Publish only build artifacts, preserving source, reports, and progress.
    for item in (ROOT/'build/html').iterdir():
        if item.is_dir():
            shutil.copytree(item, ROOT/item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, ROOT/item.name)
    if args.publish:
        run('prepare_publish.py')
        destination = (PACKAGE/'docs').resolve()
        assert destination.parent == PACKAGE.resolve() and destination.name == 'docs'
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(ROOT/'publishing/site', destination)
        print(f'Updated GitHub Pages files: {destination}')
    if args.serve:
        run('launch.py')
