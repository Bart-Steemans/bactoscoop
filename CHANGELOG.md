# Release notes

## Unreleased

- Clarify the walkthroughs, optional SVM curation, and filtering after feature extraction.
- Fix mask-saving compatibility with Omnipose 1.0.6 by requiring tifffile below 2025.2.18.

## 0.1.2

- Prevent excessive memory use when splitting cells. Add `max_daughter_cell_mesh_rows` to control daughter-mesh size (default: 800 rows).

## 0.1.1

- Improve installation instructions and documentation links.
- Remove automatic saves to developer-specific folders from plotting helpers.

## 0.1.0

Initial package, documentation, and portable example walkthroughs. This release was withdrawn from PyPI; install the latest stable release instead.
