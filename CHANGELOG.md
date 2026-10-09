# Release notes

## 0.1.2

- Reject oversized daughter meshes before contour reconstruction and dense allocations, avoiding excessive memory use during cell splitting.
- Expose `max_daughter_cell_mesh_rows` through the mesh pipeline. The default remains 800; values must be integers of at least 4. Both rebuilt daughters must satisfy the final row limits.
- Document the new parameter in the mesh workflow and API reference, separately from the feature extraction `max_mesh_size` setting.
- Install the latest stable PyPI release by default in the README and installation guide; retain tagged GitHub installs for reproducibility.
- Include daughter-mesh regression tests in release validation and update the MIT license copyright to 2026.

## 0.1.1

- Use one README for GitHub and PyPI, with links to the live documentation.
- Build distributions with patched setuptools and inspect their source, metadata and contents before publishing.
- Validate clean wheel installations, all 75 tests and both supplied-mask walkthroughs before and after PyPI publication.
- Remove automatic saves to developer-specific folders from plotting helpers.
- Refresh documentation for 0.1.1, remove the documentation review page and add personal Govers Lab attribution.

## 0.1.0

Initial packaging and documentation work. The uploaded 0.1.0 files were removed from PyPI; install the latest stable release instead.

- Added installable wheel/source packaging and Conda/uv installation guides.
- Added public documentation, automatic GitHub Pages builds, and notebook-to-documentation synchronization.
- Made walkthrough paths portable and defaulted to supplied masks with isolated working copies.
- Included tutorial plotting helpers and explicit pickle filename handling.
- Constrained SciPy and setuptools for the unpatched Omnipose 1.0.6 dependency stack.
- Removed tracked Python bytecode from the release; retained scientific example inputs and saved reference outputs.
