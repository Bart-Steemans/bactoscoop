"""Author-maintained scientific guides. Run by build.py before Sphinx."""
from pathlib import Path
import tomllib
from docs_config import PACKAGE

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
PAGES = {
'index': r'''
BactoScoop
==========

Single-cell measurements from multichannel bacterial microscopy
--------------------------------------------------------------

I developed BactoScoop to connect bacterial cell morphology with measurements
from multiple fluorescence channels. The workflow starts with microscopy images
and labeled cell masks, reconstructs a coordinate system for each cell, and
extracts measurements into a per-cell table after geometry review and curation.

Follow the guides to run that workflow, inspect intermediate results, and
understand the measurements before using them in an analysis.
Start with :doc:`getting-started/installation` and :doc:`getting-started/quickstart`,
or follow the :doc:`examples/three-channel` and :doc:`examples/five-channel` tutorials.

.. image:: _static/logo.png
   :alt: BactoScoop logo
   :width: 280px
   :align: center

.. raw:: html

   <div class="workflow" aria-label="Analysis workflow">
   <span>Images</span><b>→</b><span>Masks</span><b>→</b><span>Meshes</span><b>→</b><span>Curation</span><b>→</b><span>Objects &amp; signals</span><b>→</b><span>Feature tables</span>
   </div>

What BactoScoop measures
------------------

* **Cell morphology:** length, width, projected area, estimated volume, curvature,
  constriction, and region properties.
* **Signal profiles:** axial and mesh-based intensities, radial distributions,
  normalized profiles, and texture descriptors.
* **Detected objects:** counts, projected geometry, positions relative to the
  cell axis and poles, intensities, and estimates based on valid object meshes.
* **Membrane-associated signals:** contour profiles, gradients, and texture.
* **Relationships between channels:** implemented correlation and similarity
  measurements applied to matching feature values or profiles.

These measurements describe the supplied images and reconstructed geometry.
Biological conclusions require experimental controls and an appropriate
analysis of the resulting tables. BactoScoop's current image workflow is 2D;
volume is estimated from a 2D mesh, rather than measured from a 3D acquisition.

Two ways to learn
-----------------

The three-channel example uses phase contrast, DAPI, and GFP-DnaN to inspect
intracellular objects. The five-channel example adds outer- and inner-membrane
signals and RNA measurements. Both tutorials include real figures saved in the
repository's notebooks, complete code, and explanations of the intermediate steps.

This guide covers BactoScoop **__BACTOSCOOP_VERSION__**. Start with the installation
guide and a walkthrough, then explore the analysis workflow and API reference.

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   getting-started/installation
   getting-started/gpu
   getting-started/quickstart
   getting-started/inputs
   getting-started/outputs

.. toctree::
   :caption: Analysis workflow
   :maxdepth: 1

   workflow/segmentation
   workflow/meshes
   workflow/curation
   workflow/alignment
   workflow/objects
   workflow/features
   workflow/correlation
   workflow/export
   workflow/errors

.. toctree::
   :caption: Examples
   :maxdepth: 1

   examples/three-channel
   examples/five-channel
   examples/reuse
   examples/plotting
   examples/parallel

.. toctree::
   :caption: Reference
   :maxdepth: 1

   reference/feature-dictionary
   reference/parameters
   api/index
   reference/data-structures
   reference/troubleshooting
   reference/project
''',
'getting-started/installation': r'''
Installation
============

Install BactoScoop into a dedicated **Python 3.10** environment. Conda and uv
install the same package and dependencies. The public source repository is on
`GitHub <https://github.com/Bart-Steemans/bactoscoop>`_.

The PyPI commands require the release to be available on
`PyPI <https://pypi.org/project/bactoscoop/>`_. Until then, use the tagged
GitHub installation below.

Conda
-----

In Anaconda Prompt or Anaconda PowerShell Prompt:

.. code-block:: powershell

   conda create -n bactoscoop python=3.10 pip -y
   conda activate bactoscoop
   python -m pip install bactoscoop==__BACTOSCOOP_VERSION__
   python -m pip check
   python -c "import bactoscoop; from importlib.metadata import version; print(version('bactoscoop'))"
   jupyter lab

uv
--

After installing `uv <https://docs.astral.sh/uv/getting-started/installation/>`_:

.. code-block:: powershell

   uv venv --python 3.10 .venv
   uv pip install --python .venv bactoscoop==__BACTOSCOOP_VERSION__
   uv pip check --python .venv
   .\.venv\Scripts\python.exe -c "import bactoscoop; from importlib.metadata import version; print(version('bactoscoop'))"
   .\.venv\Scripts\jupyter.exe lab

uv downloads Python 3.10 when necessary. On macOS/Linux, use
``.venv/bin/python`` and ``.venv/bin/jupyter`` for the last two commands.

Choose a release or development source
--------------------------------------

Pin ``__BACTOSCOOP_VERSION__`` to reproduce this release. To obtain the newest stable PyPI
release, run ``python -m pip install --upgrade bactoscoop`` or
``uv pip install --python .venv --upgrade bactoscoop``.

To install the corresponding tagged GitHub release, run:

.. code-block:: powershell

   python -m pip install "bactoscoop @ git+https://github.com/Bart-Steemans/bactoscoop.git@v__BACTOSCOOP_VERSION__"

For uv, replace ``python -m pip install`` with
``uv pip install --python .venv``. Check the
`latest GitHub release <https://github.com/Bart-Steemans/bactoscoop/releases/latest>`_
for newer tags. A development checkout uses an editable install:

.. code-block:: powershell

   git clone https://github.com/Bart-Steemans/bactoscoop.git
   cd bactoscoop
   python -m pip install -e .

Dependencies
------------

.. include:: ../generated/dependencies.rst

Omnipose is pinned to 1.0.6 and NumPy remains below 2. PyArrow supplies Parquet
export.

Open the examples
-----------------

Download and extract the
`release archive <https://github.com/Bart-Steemans/bactoscoop/archive/refs/tags/v__BACTOSCOOP_VERSION__.zip>`_
for its TIFFs, masks, SVM models, and notebooks; those assets are separate from
the wheel. In JupyterLab, select the Python 3.10 analysis kernel and open a
walkthrough under ``examples/``. Its first cell copies the input dataset into
a fresh folder under ``~/bactoscoop_runs/``. Supplied masks are reused by default;
set ``RUN_SEGMENTATION=True`` to run fresh Omnipose inference.

See :doc:`quickstart` and the :doc:`../examples/three-channel` and
:doc:`../examples/five-channel` walkthroughs.
''',
'getting-started/gpu': r'''
CPU and GPU segmentation
=======================

The Omnipose wrapper calls ``cellpose_omni.core.use_gpu()`` to select GPU use.
When a usable CUDA GPU is unavailable, the normal workflow uses the CPU.
Other analysis stages use NumPy, SciPy, image processing, and plotting libraries;
the segmentation GPU setting does not move the entire pipeline onto a GPU.

Check the notebook environment
------------------------------

.. code-block:: python

   import sys
   import torch
   print(sys.executable)
   print(torch.__version__)
   print("CUDA available:", torch.cuda.is_available())

The wrapper prints ``GPU activated?`` when initialized. A first segmentation
run can require a model download. Reusing masks does not require a segmentation
model download.

Optional acceleration
---------------------

Start with the CPU installation, then choose a compatible
PyTorch build for the actual NVIDIA driver and GPU. Consult the official
`PyTorch installer <https://pytorch.org/get-started/locally/>`_ and
`Omnipose GPU guidance <https://omnipose.readthedocs.io/installation.html>`_.

An optional Windows setup uses Python 3.10, an RTX A6000, PyTorch
``2.7.1+cu118`` and torchvision ``0.22.1+cu118``. Select versions compatible
with your own GPU and driver.
Restart the notebook kernel after changing PyTorch.

If CUDA is unavailable, check the interpreter, installed PyTorch build, and
driver compatibility. A ``False`` CUDA check by itself does not prevent CPU
segmentation. See :doc:`../workflow/segmentation` for the actual wrapper settings.
''',
'getting-started/quickstart': r'''
Quick start
===========

This example reuses the supplied three-channel masks, builds meshes, curates
the cells, detects GFP-DnaN objects, and saves morphology and signal measurements.
The script first copies the dataset so that the original inputs and outputs
remain available for comparison.

Run this in the BactoScoop environment, or in its Jupyter kernel:

.. literalinclude:: ../_downloads/quickstart.py
   :language: python

Download :download:`the complete script <../_downloads/quickstart.py>`.
The default calibration matches the supplied walkthrough; replace it with your
microscope's µm/pixel calibration for your own images.

What to inspect
---------------

Inspect the mask outlines before trusting the reconstructed meshes. After
curation, compare the kept and rejected contours. Then inspect individual
object detections and check whether the merged table contains the expected
channels. See :doc:`../examples/plotting` for the plotting functions.

This quick start intentionally uses the supplied SVM with its matching dataset.
The same model need not classify cells correctly in a different experiment.
Review :doc:`../workflow/curation` before applying it to other images.

Expected files
--------------

The working copy contains new ``_meshdata.pkl``, ``_curated_meshdata.pkl``,
``_features.pkl``, and ``_features.parquet`` files. The script also saves
``processing_summary.json`` so item-level and cell-level failures can be reviewed.
Existing masks remain in the working copy's ``masks/`` folder.
''',
'getting-started/inputs': r'''
Images, channels, and masks
==========================

Organize one 2D TIFF per field and channel in a single dataset folder. The
example phase image is C1, but channel identifiers are configurable.

.. code-block:: text

   dataset/
     sample_XY1_C1.tiff
     sample_XY1_C2.tiff
     sample_XY1_C3.tiff
     sample_XY2_C1.tiff
     sample_XY2_C2.tiff
     sample_XY2_C3.tiff
     masks/
       sample_XY1_C1_cp_masks.tif
       sample_XY2_C1_cp_masks.tif

Field matching
--------------

The loaders use TIFF suffixes and natural sorting. The collection matches phase
images, masks, and channels by field identity rather than trusting array position.
The helper :py:func:`bactoscoop.utilities.image_field_key` removes the recognized
channel/mask suffix. Duplicate, missing, or mismatched identities can produce
errors. Use consistent names and one unambiguous TIFF per field and channel.

Label masks
-----------

Masks must be 2D integer label images, with background 0 and cells labeled
consecutively from 1 to N. Each matching channel and mask must have the same
height and width as the phase image. Binary foreground masks containing many
cells under label 1 do not encode distinct cells.

Generate masks through Omnipose or place externally generated masks in
``masks/`` using the matching naming convention. Inspect the overlays before
meshing. Relabel nonconsecutive labels in a separate preprocessing copy if needed.
The pipeline handles separate 2D images, not a raw multichannel stack or a 3D
volume passed directly to these loaders.

Calibration and registration
----------------------------

Pass the physical calibration into ``ImageCollection(..., px=...)``. ``px`` is
µm per pixel and defaults to 0.065. Pixel coordinates in geometry arrays are
not converted to µm; individual measurement functions apply the scale.
Check that fluorescence channels are registered to phase contrast. Per-cell
shift correction is a specific measurement operation, not a replacement for
correct acquisition geometry. See :doc:`../workflow/alignment`.
''',
'getting-started/outputs': r'''
Outputs
=======

Save geometry and measurements at different stages to inspect an analysis
or resume it without repeating every step.

.. list-table:: Common outputs
   :header-rows: 1
   :widths: 28 44 28

   * - Output
     - Contents
     - How it is produced
   * - ``masks/*_cp_masks.tif``
     - Integer segmentation labels for each selected field.
     - ``segment_images()`` via Omnipose.
   * - ``<dataset>_meshdata.pkl``
     - Cell IDs, field metadata, contours, midlines, and meshes.
     - ``batch_process_mesh(save_data=True)``.
   * - ``<dataset>_curated_meshdata.pkl``
     - Geometry for cells retained by curation.
     - ``curate_dataset(save_curated_data=True)``.
   * - ``<dataset>_meshdata.parquet``
     - Reloadable geometry stored as nested numeric lists.
     - ``meshdata_to_parquet()``.
   * - ``<dataset>_features.pkl`` / ``.parquet``
     - Merged per-cell feature table.
     - ``dataframe_to_pkl()`` / ``dataframe_to_parquet()``.
   * - ``curated_masks/*.tif`` and ``curated_mask_provenance.csv``
     - Rasterized retained contours and original-label overlap records.
     - ``export_curated_masks()``.

The default dataset name is the final component of ``image_folder_path``.
Renaming a working-copy folder therefore changes default output names.
Use explicit filenames when that matters.

Identity columns
----------------

The ordinary table uses ``image_name``, ``cell_id``, and ``frame``. A cell ID is
local to a field; use all identity columns when joining tables. ``frame`` is the
image's collection index and is not evidence of temporal tracking.
``include_metadata_tag=True`` renames these columns to ``Metadata_image_name``,
``Metadata_cell_id``, and ``Metadata_frame``.

See :doc:`../workflow/export` for naming, serialization, overwrite behavior,
and how to preserve processing summaries alongside measurements.
''',
'workflow/segmentation': r'''
Segmentation and mask review
============================

Use Omnipose to turn phase-contrast images into labeled cell masks. If masks
already exist, skip this stage and load them through ``create_image_objects()``.

.. code-block:: python

   from bactoscoop import ImageCollection
   ic = ImageCollection(r"C:\path\to\working_copy", px=0.065)
   ic.load_phase_images(phase_channel="C1")
   ok = ic.segment_images(
       mask_thresh=1, minsize=100, n=range(len(ic.images)),
       model_name="bact_phase_omni",
   )
   if not ok:
       raise RuntimeError(ic.processing_summary())
   ic.create_image_objects(phase_channel="C1")

Selection and thresholds
------------------------

``n`` selects image positions; ``None`` selects all loaded images in the wrapper.
Selected indices must be unique, valid integers. An empty selection is rejected.
The wrapper saves masks only for the selected images, so an incomplete mask set
cannot subsequently satisfy a whole-collection phase/mask match.

``minsize`` is the minimum mask size in pixels. ``mask_thresh`` becomes Omnipose's
``mask_threshold``; changing it changes how much of the predicted signal is
included. The walkthrough uses 1, but it is not a universal threshold.

The current wrapper fixes channels to ``[0, 0]``, rescale to ``None``, flow threshold
to 0.0, ``omni=True``, ``cluster=True``, ``resample=True``, ``diameter=None``,
and batch size to 1. These are wrapper implementation choices, not keyword
arguments exposed by ``ImageCollection.segment_images``.

Review the result
-----------------

.. code-block:: python

   import bactoscoop.plot as bplt
   fig = bplt.plot_segmentation_review(ic.image_objects, crop_size=500)

Look for merged cells, split cells, missing poles, and labels crossing the
image boundary before constructing meshes. A failed segmentation sets the
collection's failure state; the same collection refuses to silently reload
old masks from disk afterward. Start a fresh collection when deliberately
resuming from a known mask set.

API: :py:meth:`bactoscoop.imagecollection.ImageCollection.segment_images`,
:py:class:`bactoscoop.omni.Omnipose`, and :doc:`../getting-started/inputs`.
''',
'workflow/meshes': r'''
Contours, midlines, and meshes
=============================

BactoScoop reconstructs a cell coordinate system from each label mask so measurements
can follow the cell's shape rather than a rectangular image crop.

* A **mask** assigns an integer label to image pixels.
* A **contour** is a subpixel boundary around one cell.
* A **midline** follows the central axis through the cell.
* A **mesh** pairs points on opposing boundaries along that axis.
* A **profile mesh** supplies additional sampling coordinates for intensity
  measurements across the cell width.

Contours and midlines use ``(row, column)`` coordinates. Mesh rows hold two
opposing points in four columns; the historical names ``x1, y1, x2, y2`` do not
mean the geometry uses a Cartesian plotting convention. See
:doc:`../reference/data-structures` before passing arrays to another library.

Build cell geometry
-------------------

.. code-block:: python

   ic.batch_process_mesh(
       phase_channel="C1", join_thresh=4, split_thresh=0.5,
       CD_width=False, smoothing=0.1, save_data=True,
   )

The method joins nearby poles, creates meshes, and then splits cells when the
chosen constriction measure exceeds the threshold. ``join_thresh`` is a distance
in pixels; ``split_thresh`` is a relative constriction threshold. The signature
defaults to 0.35 for splitting, while the notebooks explicitly use 0.5.
``CD_width`` chooses the width-based constriction path when true; otherwise the
pipeline uses its phase-signal-based path. ``smoothing`` controls contour fitting.

Inspect both the overlay and population after changing these settings. Joining
and splitting change the analyzed cells and their identity; a processed cell
need not map one-to-one to an original label.

Optional neighbor filtering
---------------------------

``neighbor_filter_max_neighbors=None`` leaves the prefilter disabled. A numeric
threshold removes labels with more than that number of touching distinct labels
before expensive mesh construction. It is not a distance-based nearest-neighbor
measurement. Connectivity 1 or 4 uses shared edges; 2 or 8 also counts diagonal
contact. ``mesh_prefilter_stats`` records label counts and removal information.

Filtering crowded cells changes the sampled population. Inspect both
removal counts and representative fields before adopting it for a screen.

Geometry-based estimates
------------------------

Length integrates distances between neighboring mesh midpoints. Body width
averages the widest one-third of mesh widths. Projected area comes from the
contour polygon. Volume integrates circular cross-sections with diameter equal
to the mesh width; surface area sums a cylindrical lateral-area approximation.
These depend on a rotational geometry assumption and do not measure a 3D volume
directly. Details and units are in :doc:`../reference/feature-dictionary`.

API: :py:meth:`bactoscoop.imagecollection.ImageCollection.batch_process_mesh`.
''',
'workflow/curation': r'''
SVM curation
============

Use a trained support vector machine to remove cells whose geometry or phase
measurements resemble the rejected class in the model's training data. Curation
is a quality-control stage; the model's labels are not biological cell states.

.. code-block:: python

   import bactoscoop.plot as bplt
   before = bplt.snapshot_cell_contours(ic.image_objects)
   ic.curate_dataset(str(model_path), control=False, save_curated_data=True)
   fig = bplt.plot_curation_review(before, ic.image_objects)

The method calculates SVM features, predicts labels, retains label 1, updates
the image cell lists and mesh dataframe, and optionally writes the curated mesh
pickle. Label 0 is rejected. ``cols`` is a legacy argument and is currently unused.

Model and dataset pairing
-------------------------

The three-channel example supplies ``DnaN_Timecourse.pkl``. The five-channel
example supplies ``keio_collection_stationary.pkl``. Use each model with its
matching walkthrough and inspect the outcome before applying it elsewhere.

The model must expose ``feature_names_in_`` from training on a named pandas
DataFrame. Curation requires the actual feature columns in exactly that order.
The current dataframe preparation selects feature columns by name and sorts
them alphabetically before checking the model schema.
An unnamed model or a mismatched/reordered schema raises an error instead of
silently interpreting columns in the wrong order. Rows with nonfinite feature
values are excluded from prediction and returned as rejected rows.

State and controls
------------------

``curate_dataset`` begins with a reset of the feature tables and a maximum mesh
size of 800. Calculate your analysis features after curation. Setting
``save_curated_data=False`` suppresses the file write, but still changes the
in-memory retained cells and mesh dataframe.

``control=True`` requests randomly selected positive and negative examples.
``Curation.get_control`` actually returns ``(cell_id, frame)`` pairs and returns
``(None, None)`` if either class has too few requested examples. Some older
docstrings describe the order or empty result differently.

Export curated masks
--------------------

.. code-block:: python

   paths, provenance = ic.export_curated_masks(
       output_subfolder="curated_masks", overwrite=False,
   )

The exporter rasterizes retained contours into a separate label mask per field
and records overlaps with original labels in a CSV. It does not alter the
original segmentation masks. Rounded contours can differ from original mask
pixels; inspect the new masks. Invalid contours are skipped, and overlapping
contours are assigned in cell iteration order. Use the provenance table rather
than assuming a curated label equals the old cell ID.

API: :py:meth:`bactoscoop.imagecollection.ImageCollection.curate_dataset`,
:py:class:`bactoscoop.curation.Curation`.
''',
'workflow/alignment': r'''
Channel alignment and shift correction
======================================

Check fluorescence registration before interpreting profiles or object
positions. BactoScoop exposes several related operations with different effects.

Object alignment
----------------

``batch_detect_objects(..., align=True)`` uses the object detection alignment
path for the requested channel. The five-channel notebook enables this for
the outer-membrane channel C2 and leaves it disabled for C3–C5. This is an
example-specific choice.

Shifted geometry
----------------

``batch_shift_correction`` constructs shifted contours, midlines, and meshes.
``use_shifted_contours=True`` selects stored shifted geometry in feature
calculation. It requires the relevant shifted data to exist; enabling the flag
does not itself perform shift correction.

Signal cropping
---------------

``shift_signal=True`` activates signal handling during supported measurements.
The profiling and membrane implementations use their respective cropping and
optional-shift utilities. The ``objects`` method receives extra positional
arguments through ``*args`` and does not use ``shift_signal`` as a general
object-coordinate transformation.

Inspect these operations separately because a successful field match does
not prove subpixel registration, and changing one flag does not automatically
change every measurement. See the C2 alignment figure in
:doc:`../examples/five-channel` and the exact API signatures in
:doc:`../reference/parameters`.

.. code-block:: python

   import bactoscoop.plot as bplt
   fig = bplt.plot_C2_alignment(
       ic.image_objects[0], cell_id=43, channel="C2",
       channel_label="Outer membrane", show_objects=True,
   )

Choose a cell ID present in your own retained image before calling a cell-level
plot. The number above comes from the supplied five-channel walkthrough.
''',
'workflow/objects': r'''
Intracellular object detection
=============================

Detect signal regions within each retained cell and channel before calculating
object features. The same routine can be used for nucleoid-like regions or
puncta, but parameter choices must suit the signal and imaging conditions.

.. code-block:: python

   detections = ic.batch_detect_objects(
       channels=["C3"], reset_channels=True, align=False,
       smoothing=0.1, log_sigma=3, kernel_width=3,
       min_overlap_ratio=0.001, max_external_ratio=0.3,
   )

The operation loads and attaches channels by field identity, constructs
background-subtracted signals, thresholds filtered signals into candidate
regions, checks overlap with the cell, and attempts object geometry construction.
The returned dataframe includes field/cell information and channel detections;
full per-cell object geometry is also stored under ``cell.object_meshdata``.

Parameter choices
-----------------

``log_sigma`` controls Laplacian-of-Gaussian filtering. ``kernel_width`` affects
mask dilation; changing it changes object extent. ``min_overlap_ratio`` is the
minimum accepted overlap criterion and ``max_external_ratio`` limits external
signal. They are ratios, not distances. Refer to the linked utilities for their
actual denominators. The notebook values above differ from the method defaults.

Loading channels incrementally
-----------------------------

``reset_channels=True`` clears loaded channel dictionaries and each image's
background/interpolation caches. Use it for the first detection pass. A later
pass with ``reset_channels=False`` preserves already loaded channels, as in
the five-channel tutorial. Requested-channel detections are replaced; a failed
redetection does not preserve stale detections for that channel.

Object meshes can fail independently
-----------------------------------

A detected contour can be retained even if its mesh cannot be constructed.
The detector keeps matching empty geometry placeholders and error records.
By default, the objects feature row is omitted when required object meshes fail.
For selected channels, set
``retain_contour_on_object_mesh_failure_channels=["C4", "C5"]`` during feature
calculation to keep contour-based quantities and put NaN in mesh-dependent
aggregates. The five-channel notebook uses this option for RNA and DAPI.

No detected object and a missing feature row are not the same as an observed
count of zero. Check detections and ``cell_errors`` before interpreting missing
measurements. API:
:py:meth:`bactoscoop.imagecollection.ImageCollection.batch_detect_objects`.
''',
'workflow/features': r'''
Feature extraction
==================

Choose feature families per channel, then merge their results using the
field/cell identity. Morphology uses ``[None]`` as its channel selection.

.. code-block:: python

   ic.batch_calculate_features(
       [([None], "morphological"),
        (["C2", "C3"], "profiling"),
        (["C3"], "objects")],
       all_data=False, reset=True, max_mesh_size=1000,
   )
   features = ic.merge_dataframes(discard_morphological_nan=True)

Feature families
----------------

* ``morphological`` measures the cell geometry and inverse-phase profiles.
* ``profiling`` measures background-subtracted channel signal along or across
  the cell, plus texture.
* ``objects`` requires detected object contours for that channel.
* ``membrane`` uses detected-object/contour data to sample membrane-associated
  signal around a selected contour.
* ``svm`` supplies phase and geometry features for quality curation.

``colocalization`` exists in the source but is incomplete. The generic dispatch
can see other helper names too; only the implemented feature families above are
documented as workflow choices. The older ``phase`` mention in a docstring does
not correspond to an implemented feature method.

Profiles and texture
--------------------

Axial profiles sample around the midline. Mesh profiles average intensity across
the width at successive positions. ``normalize_per_cell`` performs min–max
normalization within that profile; a constant profile becomes zeros. This
normalization removes amplitude differences between cells and should not be
interpreted as an absolute fluorescence measurement.

Radial intensity distribution samples successive eroded contours and uses the
cropped image intensity range for normalization. GLCM descriptors use quantized
image intensities and the implementation's background masking and matrix settings.
They depend on acquisition and preprocessing as well as signal structure.

Optional data and failure behavior
---------------------------------

``all_data`` is method-specific. In the current source it gates detailed
per-object arrays for ``objects``; it does not suppress the profiling or membrane
arrays. The :doc:`../reference/feature-dictionary` identifies conditional outputs.

Cells above ``max_mesh_size`` are removed from the image's cell list. A failure
in an individual feature calculation sets its discard state for that pass and
records a cell error; the corresponding feature row is omitted. Other feature
families can still retain a row for that identity, so a merged table can contain
missing channel quantities. See :doc:`errors`.

Repeated calculations
---------------------

``reset=True`` clears all feature tables. ``reset=False`` retains other pairs,
but recalculating an existing method/channel pair replaces that pair's table.
Every feature calculation invalidates the previously merged table. Merge again
before export or downstream correlation calculations. The five-channel example
uses two passes to apply C2 signal shifting separately from C3–C5 measurements.
''',
'workflow/correlation': r'''
Signal correlation and similarity
=================================

Compare matching profiles or scalar features from different channels after
merging the feature tables. The implementation compares the supplied numerical
values; it does not establish molecular interaction or correct misregistration.

.. code-block:: python

   ic.batch_calculate_signal_correlation_features(
       ic.merged_features, ["C2", "C3"],
       [(["normalized_axial_intensity"],
         ["pearson", "spearman", "manders"])],
   )

The dataframe must contain both ``C2_<feature>`` and ``C3_<feature>``.
The helper considers channel combinations and writes columns using the
underlying function name, for example
``C2_C3_normalized_axial_intensity_pearson_correlation_coefficient``.

.. include:: ../generated/correlation-methods.rst

Interpretation and input checks
-------------------------------

Most methods require lists or arrays with more than one sample. ``ratio``
expects finite scalars and returns NaN for a zero denominator. Invalid, empty,
nonfinite, or unconvertible inputs are initially marked missing. Most paired
profile metrics require aligned arrays of equal length; their checks are not
uniform, so supply compatible one-dimensional profiles.

Constant profiles can make correlation undefined. Pearson and rank methods
may return NaN with a numerical-library warning; zero norms invalidate cosine
or overlap denominators. Distance correlation explicitly returns NaN for
constant profiles. Histogram intersection uses common bins and separately
normalized sample counts, and can compare unequal-length profiles.

The Li ICQ implementation multiplies the conventional centered fraction by 2,
giving its own -1 to 1 scale. The ``manders`` selector calculates an uncentered
overlap coefficient; it is not a pair of thresholded Manders M1/M2 fractions.
Covariance retains units of the input product. Entropy requires meaningful
nonnegative distributions; negative background-subtracted profiles need careful
interpretation. See :py:class:`bactoscoop.signalcorrelation.SignalCorrelation`.
''',
'workflow/export': r'''
Merge tables and save results
============================

Merge the feature families after completing the selected calculations.

.. code-block:: python

   import json
   from pathlib import Path
   table = ic.merge_dataframes(
       include_metadata_tag=False, discard_morphological_nan=True,
   )
   ic.dataframe_to_parquet("analysis_features.parquet")
   pickle_path = ic.dataframe_to_pkl("analysis_features.pkl")
   Path(ic.image_folder_path, "processing_summary.json").write_text(
       json.dumps(ic.processing_summary(), indent=2), encoding="utf-8",
   )

Column naming and merge behavior
--------------------------------

Stored table keys have the form ``<method>_<channel>_features``. The merge prefixes
nonidentity columns with the channel unless it is ``None``. Morphology therefore
exports ``cell_length``, whereas profiling can export ``C2_cell_length``.
Channel names containing underscores are retained.

The implementation concatenates tables and pivots on ``image_name``, ``cell_id``,
and ``frame`` with ``aggfunc="first"``. Where families emit the same channel-prefixed
column, the first nonmissing value wins; the method name is not part of the final
column prefix. Rows or columns containing only missing values can disappear under
pandas pivot behavior. ``discard_morphological_nan=True`` explicitly drops rows
missing ``cell_area`` or ``cell_length`` and requires those columns to exist.

File behavior
-------------

Feature and mesh export use temporary files and atomic replacement in the
dataset directory. Output filenames must be simple filenames, not paths outside
that folder. Existing files are replaced. ``dataframe_to_pkl`` returns the saved
path; Parquet exporters return None.

A ``.pkl`` name is used literally. A suffixless pickle name is a legacy tag:
``dataframe_to_pkl("tag")`` writes ``<dataset>_tag.pkl``. Parquet names must end in
``.parquet`` and use PyArrow with Zstandard compression and no dataframe index.
Mesh Parquet converts contour, mesh, and midline arrays to nested numeric lists
that ``batch_load_mesh`` can reload.

Save the processing summary explicitly because exporting the feature table
does not automatically save the error history or all experiment settings.
Also record calibration, channel meanings, parameter choices, model identity,
and package version for an analysis you intend to reproduce.
''',
'workflow/errors': r'''
Processing summaries and logging
================================

Check the processing summary before treating an exported table as complete.
A file can be saved even when independent fields or individual measurements
failed along the way.

.. code-block:: python

   from bactoscoop import ImageCollection, configure_bactoscoop_logging
   configure_bactoscoop_logging("WARNING")
   ic = ImageCollection("path/to/data", error_policy="continue",
                        log_cell_errors=False)
   # Run the workflow, then inspect:
   summary = ic.processing_summary()
   print(summary["status"], summary["error_count"], summary["cell_error_count"])

Two error collections
---------------------

``processing_errors`` records item/stage errors such as missing channel fields
or a whole feature pass that produces no rows. ``cell_errors`` records cell-level
feature or object-detection failures. Each record includes stage, image name,
channel, cell ID where available, error type, and message.

With ``error_policy="continue"``, independent fields can continue. Required-stage
setup failures can still raise immediately, and dependent work does not become
valid merely because continuing was requested. ``error_policy="raise"`` raises
the recorded error immediately, including cell errors.

Summary status
--------------

* ``not_started``: no successful-stage record and no processing errors.
* ``completed``: at least one stage recorded and no processing errors.
* ``completed_with_errors``: one or more processing errors exist.

The status does **not** certify that every intended analysis stage was run.
Cell errors alone do not change ``completed`` to ``completed_with_errors``.
Always inspect ``completed_stages`` and ``cell_error_count`` as well. Records
accumulate within the collection; a summary is not a fresh per-call report.

Logging levels
--------------

``configure_bactoscoop_logging`` updates the logger and ``BACTOSCOOP_LOG_LEVEL``.
The default is INFO. WARNING reduces routine stage messages and disables the
package's conditional tqdm bars. ``log_cell_errors=True`` prints each cell error;
errors remain recorded even when that option is false. Some older print calls
and external-library warnings can still appear independently of the logger.
''',
'examples/reuse': r'''
Reuse masks and saved meshes
===========================

Resume at different points in the workflow, provided the inputs and saved
geometry belong to the same fields and calibration.

Reuse existing masks
--------------------

.. code-block:: python

   from bactoscoop import ImageCollection
   ic = ImageCollection(r"C:\path\to\data", px=0.065)
   ic.create_image_objects(phase_channel="C1")
   ic.batch_process_mesh(phase_channel="C1", save_data=True)

``create_image_objects`` loads missing phase images and masks, validates their
field identities, and creates new image objects. In the tutorials, set
``RUN_SEGMENTATION=False`` to take this route.

Reuse geometry
--------------

.. code-block:: python

   ic = ImageCollection(r"C:\path\to\data", px=0.065)
   ic.batch_load_mesh("data_curated_meshdata.pkl", phase_channel="C1")
   # Continue with channel detection and feature extraction.

``batch_load_mesh`` accepts pickle and reloadable mesh Parquet. ``pkl_path`` can
specify the directory containing the mesh file; its default is the image folder.
The image/mask files are still needed to reconstruct image objects and measure
signals. The file needs compatible geometry columns and field identities.

State after loading
-------------------

Reloading creates a new image/cell population and clears previous feature and
merged tables. Saved mesh tables preserve geometry, not all channel caches,
detected objects, or feature dictionaries. Redetect objects and calculate the
requested features after reloading.

Verify ``px`` against the original experiment rather than assuming the saved
geometry file carries all acquisition settings. See
:py:meth:`bactoscoop.imagecollection.ImageCollection.batch_load_mesh`.
''',
'examples/plotting': r'''
Plotting and visual quality control
==================================

Use field views to understand the input and cell views to check the geometry
behind measurements. The walkthrough figures are linked from
:doc:`three-channel` and :doc:`five-channel`.

.. code-block:: python

   import bactoscoop.plot as bplt
   field_ids = bplt.list_field_ids(ic.image_folder_path, phase_channel="C1")
   fig = bplt.plot_field_channels(
       ic.image_folder_path, field_ids[0],
       {"C1": "Phase contrast", "C2": "DAPI", "C3": "GFP-DnaN"},
   )
   fig = bplt.plot_segmentation_review(ic.image_objects)
   image = ic.image_objects[0]
   cell_id = image.cells[0].cell_id
   fig = bplt.plot_cell_gallery(image, cell_id, channel=None)

Select a cell after the relevant stage
------------------------------------

Curation and mesh filtering can remove a selected cell. Some field/cell choices
in saved notebooks are example-specific. Use ``select_example_cell`` or
``pick_representative_cell`` to inspect available cells and verify the return
value against its :doc:`../api/plot` entry. ``pick_detected_cell`` chooses a cell
with requested object detections when possible.

Signals and detections
----------------------

.. code-block:: python

   cell_id = bplt.pick_detected_cell(image, preferred_channels=("C3",))
   fig = bplt.plot_signal_gallery(
       image, cell_id, channel_labels={"C3": "GFP-DnaN"}, show_objects=True,
   )
   fig = bplt.plot_feature_overview(ic.merged_features, signal_column="C3_NC_ratio")

Additional functions include histograms, feature scatter plots, morphology and
signal summaries, normalized axial profiles, demographs, and pairwise metric
heatmaps. See :doc:`../api/plot` for signatures and return shapes. Some functions
display directly, while others return a figure or a collection of figures.
In notebooks, explicitly display returned figures and close them after use
when generating many plots.

Distinguish display contrast from quantitative preprocessing: making a channel
look brighter in a figure does not change its recorded feature values.
''',
'examples/parallel': r'''
Parallel processing of sample folders
====================================

The repository includes a screening script in
``examples/BactoScoop Parallel Processing.py`` and three drive-specific variants
at the repository root. They show how to apply the single-folder workflow to
many samples. Their acquisition drives, model paths, worker count, and channel
choices describe a specific experiment.

First validate one representative folder serially, then distribute
independent folders among workers. Keep each worker's outputs within its own
sample folder to avoid competing writes.

.. code-block:: python

   # Set these before importing NumPy or BactoScoop in worker processes.
   import os
   for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
       os.environ.setdefault(variable, "1")
   os.environ.setdefault("BACTOSCOOP_LOG_LEVEL", "WARNING")

   from multiprocessing import get_context

   def process_folder(folder):
       from bactoscoop import ImageCollection
       ic = ImageCollection(folder, px=0.065)
       ic.create_image_objects(phase_channel="C1")
       ic.batch_process_mesh(phase_channel="C1", save_data=True)
       return folder, ic.processing_summary()

   if __name__ == "__main__":
       folders = [r"C:\data\sample_A", r"C:\data\sample_B"]
       with get_context("spawn").Pool(processes=2, maxtasksperchild=4) as pool:
           for folder, summary in pool.imap_unordered(process_folder, folders):
               print(folder, summary["status"])

This minimal example processes meshes from existing masks. Add curation,
detection, measurement, and export only after confirming the intended settings.
Use a script with a ``__main__`` guard on Windows, rather than pasting the worker
example directly into a notebook.

Memory and progress
-------------------

The screening scripts limit numerical-library threads, recycle workers with
``MAX_TASKS_PER_CHILD``, and record stage/progress information. Choose the worker
count for available memory and workload; the script's default 12 cores is not a
recommendation for every machine. Multiple simultaneous GPU segmentation jobs
can compete for GPU memory.

The full screening scripts were inspected, not executed for this documentation.
The documentation does not claim a newly measured throughput benchmark.
''',
'reference/data-structures': r'''
Data structures and coordinate conventions
==========================================

Collection, image, and cell
--------------------------

``ImageCollection`` holds a dataset path and calibration, loaded phase images,
masks, channel dictionaries, image objects, geometry tables, feature tables,
and processing records. ``Image`` holds a single field and its cells, channel
data, background/interpolation caches, and stage-specific tables. ``Cell`` holds
one cell's contour, midline, mesh, detected-object data, feature dictionaries,
and discard state.

.. list-table:: Geometry representation
   :header-rows: 1

   * - Object
     - Shape / convention
   * - Image and mask
     - ``(height, width)``; array indexing is row then column.
   * - Contour and midline
     - ``(N, 2)``; points use ``(row, column)`` in pixel units.
   * - Cell mesh
     - ``(N, 4)``; two opposing boundary points per row.
   * - Object geometry
     - Lists of contours, meshes, and midlines per channel; failed meshes may be empty arrays.
   * - Profile feature
     - A one-dimensional array or list stored in a dataframe cell.

The historical mesh attributes ``x1``, ``y1``, ``x2``, and ``y2`` are slices of
the stored geometry. OpenCV drawing needs Cartesian x/y points, so the curated
mask exporter explicitly reverses contour columns. Plot overlays also adapt
the convention. Do not reverse arrays solely because an attribute says ``x``.

Object dictionary
-----------------

``cell.object_meshdata[channel]`` contains detected ``object_contour``,
``object_mesh``, and ``object_midline`` lists, with construction-error information
where applicable. Their entries remain aligned even when a mesh fails.

Feature dictionaries
--------------------

The methods populate ``morphological_features``, ``profiling_features``,
``objects_features``, ``membrane_features``, and ``svm_features`` on cells.
``profiling_data`` stores intermediate data, including widths used during
morphology, and is not automatically another exported feature family.

Collection tables
-----------------

``mesh_df_collection`` stores combined geometry. ``feature_dataframes`` holds
separate family/channel tables. ``merged_features`` is created by merging;
it is invalidated by later feature computation or image population recreation.
Do not export an earlier merged table after changing the calculation state.

``cell_id`` and ``frame`` are analysis identities. They are not lineage IDs or
evidence that the package tracks cells through time.
''',
'reference/troubleshooting': r'''
Troubleshooting
===============

Import or kernel problems
-------------------------

Check ``sys.executable`` in the notebook, then install the package in that
interpreter's Python 3.10 environment. Use ``python -m pip check`` to inspect
dependency conflicts. A documentation build uses a separate interpreter and
does not need to import the segmentation stack.

No images, unmatched masks, or missing channels
---------------------------------------------

Check the dataset folder, TIFF extension, selected phase suffix, and field names.
Compare image dimensions. Masks must be integer and consecutively labeled.
Duplicate field identities are rejected. Missing channels are recorded rather
than silently matched to another field. See :doc:`../getting-started/inputs`.

Segmentation failure
--------------------

Check the returned boolean and ``processing_summary`` before making image objects.
An existing mask on disk does not automatically rescue a failed segmentation
in the same collection. Confirm model availability and CPU/GPU setup; restart
with a fresh collection only when intentionally reusing verified masks.

Unexpected geometry or too few cells
-----------------------------------

Inspect masks, join/split settings, smoothing, and neighbor prefilter statistics.
Check large-mesh filtering and SVM rejection counts. A saved table reflects
retained cells, not necessarily every original label.

SVM schema error
----------------

Use the matching supplied model or train a model on the named feature dataframe.
The model must carry ``feature_names_in_`` and its order must match. Do not bypass
the schema guard by guessing a column order.

Missing object features
-----------------------

Check whether detection actually found contours for the cell and channel.
Check object mesh errors and the retention option for contour-only measurements.
Missing rows must not automatically be counted as zero objects. Review
``cell_errors`` even when summary status says ``completed``.

Export fails or contains unexpected columns
------------------------------------------

Recalculate and merge in the intended order. ``reset=True`` discards prior tables;
``reset=False`` replaces repeated pairs. A merge can lose entirely missing columns,
and shared channel/feature names use the first nonmissing value. A filename must
stay within the image folder, and a Parquet name must end in ``.parquet``.

Plots cannot find a cell
-----------------------

The selected cell may have been removed during processing or curation. Select
from the current ``image.cells`` list and ensure the relevant channel/detections
exist. Saved notebook cell IDs are illustrative, not stable for new settings.
''',
'reference/project': r'''
Project information
===================

Author and attribution
----------------------

I'm Bart Steemans from the `Govers Lab <https://www.goverslab.com/>`_ at KU Leuven.
I developed BactoScoop for image-based profiling of individual bacterial cells.
Segmentation uses Omnipose and its Cellpose fork; the source links to those
dependencies and the current package metadata pins Omnipose 1.0.6.
Please acknowledge the original tools appropriately in work based on their use.

Version and changelog
---------------------

BactoScoop **__BACTOSCOOP_VERSION__** supports **Python 3.10**. Install it from PyPI
or download a tagged release from GitHub. See the changelog below for release notes.

.. literalinclude:: ../_downloads/CHANGELOG.md
   :language: text

Project links
-------------

Find the documentation, public source, releases, and issue tracker here:

* `Documentation <https://bart-steemans.github.io/bactoscoop/>`_
* `Releases <https://github.com/Bart-Steemans/bactoscoop/releases>`_
* `PyPI <https://pypi.org/project/bactoscoop/>`_
* `Repository <https://github.com/Bart-Steemans/bactoscoop>`_
* `Issue tracker <https://github.com/Bart-Steemans/bactoscoop/issues>`_
* `Omnipose documentation <https://omnipose.readthedocs.io/>`_

License
-------

BactoScoop is distributed under the MIT license. The source license identifies
copyright © 2023 Bart Steemans.

.. literalinclude:: ../_downloads/LICENSE
   :language: text

Documentation design
--------------------

The documentation follows the scientific layout used by the Omnipose site, with
Sphinx and Furo for navigation, search, and light/dark appearance. The BactoScoop
content is based on this repository. Required scripts, styles, fonts supplied
by the theme, and figures are local assets; reading the built site does not
require a network connection.
''',
}

def write_guides():
    # Retire the old generated page when rebuilding an existing local checkout.
    retired = (SOURCE / 'reference/provenance.rst').resolve()
    assert retired.is_relative_to(SOURCE.resolve())
    if retired.is_file():
        retired.unlink()
    release = tomllib.loads((PACKAGE/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    for slug, content in PAGES.items():
        content = content.replace('__BACTOSCOOP_VERSION__',release)
        lines = content.strip().splitlines()
        for i in range(1, len(lines)):
            if lines[i] and lines[i][0] in '=~-^' and len(set(lines[i])) == 1 and not lines[i-1].startswith(' '):
                lines[i] = lines[i][0] * len(lines[i-1])
        content = '\n'.join(lines)
        path = SOURCE / f'{slug}.rst'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + '\n', encoding='utf-8')

if __name__ == '__main__':
    write_guides()
