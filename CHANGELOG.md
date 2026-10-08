# Release notes

## 0.1.1

- Use one README for GitHub and PyPI, with links to the live documentation.
- Build distributions with patched setuptools and inspect their source, metadata and contents before publishing.
- Validate clean wheel installations, all 75 tests and both supplied-mask walkthroughs before and after PyPI publication.
- Remove automatic saves to developer-specific folders from plotting helpers.
- Refresh documentation for 0.1.1, remove the documentation review page and add personal Govers Lab attribution.

## 0.1.0

Initial packaging and documentation work. The uploaded 0.1.0 files were removed from PyPI; use 0.1.1.

- Added installable wheel/source packaging and Conda/uv installation guides.
- Added public documentation, automatic GitHub Pages builds, and notebook-to-documentation synchronization.
- Made walkthrough paths portable and defaulted to supplied masks with isolated working copies.
- Included tutorial plotting helpers and explicit pickle filename handling.
- Constrained SciPy and setuptools for the unpatched Omnipose 1.0.6 dependency stack.
- Removed tracked Python bytecode from the release; retained scientific example inputs and saved reference outputs.
