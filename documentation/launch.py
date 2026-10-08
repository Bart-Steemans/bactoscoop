"""Serve the rendered documentation locally and open it in the default browser."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
import webbrowser
import urllib.request

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--no-browser', action='store_true')
args = parser.parse_args()
site = ROOT/'build/html' if (ROOT/'build/html/index.html').is_file() else ROOT
address = f'http://127.0.0.1:{args.port}/'
# Check before binding: Windows can otherwise allow multiple listeners when
# a server enables SO_REUSEADDR.
try:
    with urllib.request.urlopen(address, timeout=2) as response:
        existing = response.read(20000).decode('utf-8',errors='replace')
    if '<title>BactoScoop' in existing:
        print(f'BactoScoop documentation is already running: {address}')
        if not args.no_browser:
            webbrowser.open(address)
        raise SystemExit(0)
except (OSError, urllib.error.URLError):
    pass
try:
    server = ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(site)))
except OSError as error:
    address = f'http://127.0.0.1:{args.port}/'
    try:
        with urllib.request.urlopen(address, timeout=2) as response:
            existing = response.read(20000).decode('utf-8',errors='replace')
        if '<title>BactoScoop' in existing:
            print(f'BactoScoop documentation is already running: {address}')
            if not args.no_browser:
                webbrowser.open(address)
            raise SystemExit(0)
    except (OSError, urllib.error.URLError):
        pass
    raise SystemExit(f'Cannot use port {args.port}: {error}. Try --port 8001.')
address = f'http://127.0.0.1:{args.port}/'
print(f'BactoScoop documentation: {address}\nPress Ctrl+C to stop.')
if not args.no_browser:
    Timer(0.5,lambda:webbrowser.open(address)).start()
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
