project = 'BactoScoop'
author = 'Bart Steemans'
copyright = '2026, Bart Steemans'
import sys
from pathlib import Path
import tomllib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docs_config import PACKAGE
version = release = tomllib.loads((PACKAGE/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
extensions = ['sphinx_copybutton', 'myst_parser']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
exclude_patterns = ['generated/**', '_downloads/**']
html_theme = 'furo'
html_title = 'BactoScoop documentation'
html_logo = '_static/logo.png'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'sidebar_hide_name': False,
    'light_css_variables': {'color-brand-primary': '#16766c', 'color-brand-content': '#12685f'},
    'dark_css_variables': {'color-brand-primary': '#78d8bc', 'color-brand-content': '#78d8bc'},
}
html_show_sourcelink = True
html_copy_source = True
html_show_sphinx = False
html_domain_indices = True
html_search_language = 'en'
pygments_style = 'sphinx'
pygments_dark_style = 'monokai'
copybutton_prompt_text = r'>>> |\.\.\. |\$ |PS> '
copybutton_prompt_is_regexp = True
myst_enable_extensions = ['colon_fence']
myst_heading_anchors = 3
nitpicky = True
