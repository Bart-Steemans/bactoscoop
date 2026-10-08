"""One-command documentation environment setup and notebook/source synchronization."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
environment = PACKAGE/'.docs-venv'
interpreter = environment/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
uv = shutil.which('uv')
if not interpreter.is_file():
    if uv:
        subprocess.run([uv,'venv','--python','3.11',str(environment)],check=True)
        subprocess.run([uv,'pip','install','--python',str(interpreter),'-r',str(ROOT/'requirements-docs.txt')],check=True)
    else:
        if sys.version_info < (3,11):
            raise SystemExit('Install uv or run this command with Python 3.11 or newer to create the documentation environment.')
        subprocess.run([sys.executable,'-m','venv',str(environment)],check=True)
        subprocess.run([str(interpreter),'-m','pip','install','-r',str(ROOT/'requirements-docs.txt')],check=True)
subprocess.run([str(interpreter),str(ROOT/'build.py'),'--publish',*sys.argv[1:]],cwd=PACKAGE,check=True)
