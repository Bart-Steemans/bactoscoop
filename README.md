# BactoScoop

<p align="center"><img src="https://raw.githubusercontent.com/Bart-Steemans/bactoscoop/main/BactoScoop%20Logo.png" alt="BactoScoop logo" width="310"></p>

**BactoScoop enables reproducible high-throughput image-based profiling of individual bacterial cells.** It takes multichannel microscopy images from individual fields through segmentation, cell mesh construction, quality curation, object detection, and feature extraction. The result is a per-cell table that connects morphology with membrane, nucleoid, and other fluorescence measurements. Each stage can be inspected, and saved masks, meshes, and feature tables make an analysis easier to review and repeat.
[**Read the documentation**](https://bart-steemans.github.io/bactoscoop/) · [Three-channel walkthrough](https://bart-steemans.github.io/bactoscoop/examples/three-channel.html) · [Five-channel walkthrough](https://bart-steemans.github.io/bactoscoop/examples/five-channel.html)

## Install BactoScoop

BactoScoop requires **Python 3.10**. Choose either Conda or uv; both install the same package and its dependencies. The segmentation stack includes Omnipose 1.0.6 and NumPy below 2.

Install BactoScoop from [PyPI](https://pypi.org/project/bactoscoop/) using either environment option below. A tagged GitHub installation is also available.

### Conda

Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) if needed, then run in Anaconda Prompt or Anaconda PowerShell Prompt:

```powershell
conda create -n bactoscoop python=3.10 pip -y
conda activate bactoscoop
pip install bactoscoop
python -m pip check
python -c "import bactoscoop; from importlib.metadata import version; print(version('bactoscoop'))"
jupyter lab
```

### uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), open PowerShell in a folder for your analysis, then run:

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv bactoscoop
uv pip check --python .venv
.\.venv\Scripts\python.exe -c "import bactoscoop; from importlib.metadata import version; print(version('bactoscoop'))"
.\.venv\Scripts\jupyter.exe lab
```

uv can download Python 3.10 if needed. On macOS/Linux, use `.venv/bin/python` and `.venv/bin/jupyter` for the last two commands.

### Stable release, GitHub release, or development checkout

The commands above install the latest stable PyPI release in a new environment. To upgrade an existing installation, use `python -m pip install --upgrade bactoscoop` (Conda) or `uv pip install --python .venv --upgrade bactoscoop` (uv).

The corresponding [GitHub release](https://github.com/Bart-Steemans/bactoscoop/releases/latest) can also be installed directly when Git is installed:

```powershell
python -m pip install "bactoscoop @ git+https://github.com/Bart-Steemans/bactoscoop.git@v0.1.2"
```

For uv, use `uv pip install --python .venv "bactoscoop @ git+https://github.com/Bart-Steemans/bactoscoop.git@v0.1.2"`. Select the tag shown in Releases when a newer release is available. To work on the development code instead:

```powershell
git clone https://github.com/Bart-Steemans/bactoscoop.git
cd bactoscoop
python -m pip install -e .
```

An editable development install follows changes in your checkout. It may differ from the stable release. All installation commands must use your Python 3.10 analysis environment. Select that environment's kernel in JupyterLab.

## Run a walkthrough

The wheel contains the Python package; the large TIFFs, masks, SVM models, and notebooks are supplied separately. Download and extract the [v0.1.2 example archive](https://github.com/Bart-Steemans/bactoscoop/archive/refs/tags/v0.1.2.zip), or use the `examples/` folder in a cloned checkout.

Open `examples/3_channel_example_walkthrough.ipynb` or `examples/5_channel_example_walkthrough.ipynb` in JupyterLab. Run cells from top to bottom. The first cell locates the downloaded examples and copies the selected dataset into a fresh folder under `~/bactoscoop_runs/`; the supplied images, masks, and saved reference outputs stay intact. If you open the notebook elsewhere, set the `BACTOSCOOP_EXAMPLES_DIR` environment variable to the downloaded `examples` folder before starting JupyterLab.

The examples reuse the supplied masks by default. Set `RUN_SEGMENTATION = True` for fresh Omnipose inference; the first inference may download model weights. Set `PIXEL_SIZE_UM` to your microscope calibration. The first cell prints the working output folder. Meshes, curated meshes, and feature tables are saved there. The example-specific SVM models are copied with their datasets; inspect their predictions before using those models on different images.

## Segmentation on CPU or GPU

Segmentation **automatically runs on the CPU** when a usable CUDA GPU is unavailable. No additional configuration is required for the standard installation.

If CUDA-enabled PyTorch and a compatible NVIDIA GPU are available, Omnipose can use GPU acceleration. The walkthroughs include a check to verify whether PyTorch detects CUDA.

### Optional GPU acceleration

To enable GPU acceleration, first install BactoScoop following the standard installation instructions. Then:

1. **Check GPU compatibility:** Consult the [Omnipose documentation](https://omnipose.readthedocs.io/installation.html) for GPU support, requirements, and Python compatibility.

2. **Install CUDA-enabled PyTorch:** Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) to select an appropriate CUDA-enabled build for your GPU and NVIDIA driver. Install the corresponding PyTorch and torchvision versions in your BactoScoop environment.

   For reference, we have successfully tested segmentation on Windows with Python 3.10, an NVIDIA RTX A6000, PyTorch 2.7.1+cu118, and torchvision 0.22.1+cu118, installed using:

   ```powershell
   python -m pip install --upgrade torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
   ```

   This is an example of a working configuration, not a requirement. Other supported versions can be found in the [PyTorch previous versions documentation](https://pytorch.org/get-started/previous-versions/).

3. **Verify CUDA availability:** Run the following command in the same environment:

   ```powershell
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

   If the output is `CUDA available: True`, PyTorch can access CUDA. If it returns `False`, segmentation will still run on the CPU.

**Important:** Restart your Jupyter kernel after installing or updating PyTorch.

## Expected input images (and masks)

Place one 2D TIFF per channel and field in the same folder. The shared name before `_C1`, `_C2`, and so on identifies a field. `C1` is the phase image in the examples.

```text
your_images/
  sample_XY1_C1.tiff
  sample_XY1_C2.tiff
  sample_XY1_C3.tiff
  sample_XY2_C1.tiff
  sample_XY2_C2.tiff
  sample_XY2_C3.tiff
  masks/
    sample_XY1_C1_cp_masks.tif
    sample_XY2_C1_cp_masks.tif
```

Channels and masks for one field must have the same height and width. BactoScoop can segment the phase image with Omnipose or use masks made by another program. Masks must be 2D integer images with background **0** and distinct cells labeled consecutively **1 through N**. Check mask outlines before building meshes. Set the correct pixel size in µm/pixel and verify channel registration when measuring signals across channels.

## Walkthroughs and the analysis stages

| Walkthrough | Channels | Main example |
| --- | --- | --- |
| [Three-channel](https://github.com/Bart-Steemans/bactoscoop/blob/main/examples/3_channel_example_walkthrough.ipynb) | `C1` phase; `C2` DAPI, showing DNA/nucleoid morphology; `C3` GFP-DnaN, showing replication-associated puncta. | Detect intracellular objects such as the nucleoid and DnaN foci and extract morphology and object related features. The supplied SVM is `DnaN_Timecourse.pkl`. |
| [Five-channel](https://github.com/Bart-Steemans/bactoscoop/blob/main/examples/5_channel_example_walkthrough.ipynb) | `C1` phase; `C2` outer membrane; `C3` inner membrane; `C4` RNA; `C5` DAPI/nucleoid. | This tutorial performs BactoScoop on a 5 channel set and includes additional extraction of membrane features. The supplied SVM is `keio_collection_stationary.pkl`. |

An example script of how we used BactoScoop to screen thousands of bacterial samples is also available in the example folder (named "BactoScoop Parallel Processing.py").

Load pickle-based SVM models and saved mesh tables only from trusted sources:
pickle files can execute code when loaded. Use Parquet to share feature tables.
