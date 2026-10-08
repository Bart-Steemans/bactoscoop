Release validation — 2026-10-07
--------------------------------

The 0.1.0 wheel and source distribution passed strict Twine checks. The wheel contains package code and metadata, without caches or large example datasets.

Validation used separate Python 3.10.20 Conda and uv environments. Imports resolve to their installed site-packages rather than the source checkout. Dependency checks pass.

* **uv:** 19 safeguard tests; 0 failures/errors.
  ``3_channel_example_walkthrough.ipynb``: 130 rows, 188 columns; 0 processing errors and 0 cell-level failure records.
  ``5_channel_example_walkthrough.ipynb``: 227 rows, 390 columns; 0 processing errors and 35 cell-level failure records.
* **conda:** 19 safeguard tests; 0 failures/errors.
  ``3_channel_example_walkthrough.ipynb``: 129 rows, 188 columns; 0 processing errors and 0 cell-level failure records.
  ``5_channel_example_walkthrough.ipynb``: 228 rows, 390 columns; 0 processing errors and 39 cell-level failure records.

These executions reuse supplied masks; fresh Omnipose inference and GPU benchmarks were not performed. The five-channel run records cells with "No object contours to calculate features from"; do not interpret those missing measurements as zero detections. Cell retention can vary slightly between runs. Each run's output and processing summary were retained.

The documentation passes its local-link and browser checks. An isolated portable checkout verifies that edited notebook markdown reaches the website and downloadable notebook, and that all ten saved five-channel PNGs match the source notebook. Original inputs stay unchanged during validation.

These checks validate the local wheel; they do not establish availability or successful installation from PyPI.
