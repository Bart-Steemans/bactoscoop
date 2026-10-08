"""Validate generated links, inventory coverage, and browser interaction offline."""
import ast
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import Thread
from urllib.parse import unquote, urlsplit
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
SITE = ROOT/'build/html'
REPORTS = ROOT/'reports'

def static_checks():
    pages = list(SITE.rglob('*.html'))
    soups = {p:BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser') for p in pages}
    anchors = {p:{tag['id'] for tag in soup.find_all(id=True)} for p,soup in soups.items()}
    failures = []
    checked = 0
    remote_assets = []
    for path,soup in soups.items():
        for tag in soup.find_all(['a','img','script','link']):
            attribute = 'src' if tag.name in ['img','script'] else 'href'
            value = tag.get(attribute)
            if not value:
                continue
            url = urlsplit(value)
            if url.scheme or url.netloc:
                if tag.name!='a' and url.scheme in ['http','https']:
                    remote_assets.append({'page':str(path.relative_to(SITE)),'asset':value})
                continue
            if not url.path and not url.fragment:
                continue
            target = (SITE/url.path.lstrip('/')) if url.path.startswith('/') else (path.parent/unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target = target/'index.html'
            checked += 1
            if not target.exists():
                failures.append({'page':str(path.relative_to(SITE)),'target':value,'reason':'missing file'})
            elif url.fragment and target.suffix=='.html' and unquote(url.fragment) not in anchors.get(target,set()):
                failures.append({'page':str(path.relative_to(SITE)),'target':value,'reason':'missing fragment'})
    coverage = json.loads((REPORTS/'api-coverage.json').read_text(encoding='utf-8'))
    feature_coverage = json.loads((REPORTS/'feature-coverage.json').read_text(encoding='utf-8'))
    inspection = json.loads((REPORTS/'inspection.json').read_text(encoding='utf-8'))
    expected = {(a['module'],(a['class']+'.' if a['class'] else '')+a['name']) for a in inspection['api'] if a['module']!='__init__' and a['name']!='__init__'}
    actual = {(a['module'],a['name']) for a in coverage}
    missing_api = sorted(expected-actual)
    feature_pairs = {(f['category'],f['key']) for f in feature_coverage}
    missing_features = sorted({(f['category'],f['key']) for f in inspection['features']}-feature_pairs)
    syntax = []
    for path in (ROOT/'source/_downloads').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
        syntax.append(path.name)
    for path in (ROOT/'source/_downloads').glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                source = '\n'.join(line for line in ''.join(cell['source']).splitlines() if not line.lstrip().startswith(('%','!')))
                ast.parse(source)
        syntax.append(path.name)
    return {'html_pages':len(pages),'checked_local_links':checked,'broken_links':failures,'remote_required_assets':remote_assets,'documented_api_entries':len(coverage),'missing_api':missing_api,'documented_feature_columns':len(feature_coverage),'missing_features':missing_features,'download_syntax_passed':syntax}

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self,*args):
        pass

def browser_checks():
    from playwright.sync_api import sync_playwright
    server = ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(SITE)))
    Thread(target=server.serve_forever,daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'
    results = {'base':'local ephemeral HTTP server','pages':[],'console_errors':[],'failed_requests':[]}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
            context = browser.new_context(viewport={'width':1440,'height':1000},permissions=['clipboard-read','clipboard-write'])
            page = context.new_page()
            page.set_default_timeout(10000)
            page.on('pageerror',lambda error:results['console_errors'].append(str(error)))
            page.on('requestfailed',lambda request:results['failed_requests'].append({'url':request.url,'failure':request.failure}))
            remote = []
            def route(request_route):
                if not request_route.request.url.startswith(base):
                    remote.append(request_route.request.url)
                    request_route.abort()
                else:
                    request_route.continue_()
            context.route('**/*',route)
            screenshots = ROOT/'validation/screenshots'
            screenshots.mkdir(parents=True,exist_ok=True)
            for relative in ['index.html','examples/three-channel.html','examples/five-channel.html','reference/features-morphological.html','reference/project.html','api/imagecollection.html']:
                page.goto(base+'/'+relative,wait_until='networkidle')
                title = page.locator('h1').first.inner_text()
                loaded = page.locator('img').evaluate_all('(images) => images.every(i => i.complete && i.naturalWidth > 0)')
                overflow = page.evaluate('document.documentElement.scrollWidth > window.innerWidth + 2')
                results['pages'].append({'page':relative,'h1':title,'images_loaded':loaded,'horizontal_overflow':overflow})
            page.goto(base+'/index.html',wait_until='networkidle')
            page.screenshot(path=str(screenshots/'desktop-home.png'))
            before = page.evaluate('getComputedStyle(document.body).backgroundColor')
            after = before
            for _ in range(3):
                page.locator('button.theme-toggle:visible').first.click()
                page.wait_for_timeout(150)
                after = page.evaluate('getComputedStyle(document.body).backgroundColor')
                if after != before:
                    break
            results['theme_toggle_changes_background'] = before != after
            page.screenshot(path=str(screenshots/'desktop-dark.png'))
            page.goto(base+'/getting-started/quickstart.html',wait_until='networkidle')
            page.locator('button.copybtn').first.click()
            page.wait_for_timeout(150)
            clipboard = page.evaluate('navigator.clipboard.readText()')
            results['code_copy_works'] = 'ImageCollection' in clipboard
            page.goto(base+'/search.html?q=constriction',wait_until='networkidle')
            page.locator('#search-results li').first.wait_for(timeout=15000)
            results['search_results'] = page.locator('#search-results li').count()
            page.screenshot(path=str(screenshots/'search-results.png'))
            page.set_viewport_size({'width':390,'height':844})
            page.goto(base+'/index.html',wait_until='networkidle')
            results['mobile_horizontal_overflow'] = page.evaluate('document.documentElement.scrollWidth > window.innerWidth + 2')
            page.screenshot(path=str(screenshots/'mobile-home.png'))
            toggles = page.locator('label.nav-overlay-icon[for="__navigation"]')
            if toggles.count():
                toggles.first.click()
                results['mobile_navigation_opens'] = page.locator('#__navigation').is_checked()
            else:
                results['mobile_navigation_opens'] = False
            results['remote_requests'] = remote
            browser.close()
    except Exception as error:
        results['error'] = str(error)
    finally:
        server.shutdown()
        server.server_close()
    return results

def main():
    result = {'static':static_checks(),'browser':browser_checks()}
    (REPORTS/'site-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    static = result['static']
    browser = result['browser']
    bad = any(static[k] for k in ['broken_links','remote_required_assets','missing_api','missing_features'])
    bad |= 'error' in browser or bool(browser.get('console_errors')) or bool(browser.get('failed_requests')) or bool(browser.get('remote_requests'))
    bad |= not browser.get('theme_toggle_changes_background') or not browser.get('code_copy_works') or not browser.get('search_results') or not browser.get('mobile_navigation_opens')
    bad |= any(not p['images_loaded'] or p['horizontal_overflow'] for p in browser.get('pages',[])) or bool(browser.get('mobile_horizontal_overflow'))
    raise SystemExit(1 if bad else 0)

if __name__=='__main__':
    main()
