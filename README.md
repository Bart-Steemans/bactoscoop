# BactoScoop: from multichannel micrographs of bacteria to quantifiable single-cell features

<p align="center">
  <img src="https://github.com/Bart-Steemans/bactoscoop/blob/main/logo.jpg?raw=true" alt="BactoScoop" width="300"/>
</p>

This application is built to extract as many single cell features as possible from multichannel microscopy images of bacteria. It uses Omnipose (https://doi.org/10.1038/s41592-022-01639-4) to segment phase contrast images, creates an internal coordinate system (or cell mesh) for each cell where numerous features are extracted from (related to shapes, sizes, objects, intensity, texture, etc.). If you train a support vector machine model, there is also the possibility of omitting poor segmentation masks (https://doi.org/10.1016/j.xpro.2024.102868). 

Disclaimer: the program is still under development, as such, any issues, comments, or ideas are more than welcome!

# How to install BactoScoop

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path. Alternatives like miniconda also work just as well.

2. Open an anaconda prompt / command prompt with conda for python 3 in the path.

3. To create a new environment for CPU only, run
```
conda create -n bactoscoop 'python==3.10.12' pytorch
```
4. For users that want to accelerate Omnipose segmentation with NVIDIA GPUs, add these additional arguments:
```
torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
See [GPU support](https://github.com/kevinjohncutler/omnipose?tab=readme-ov-file#gpu-support) for more details. See [Python compatibility](https://github.com/kevinjohncutler/omnipose?tab=readme-ov-file#python-compatibility) for more about choosing your python version.

5. To activate this new environment, run
```
conda activate bactoscoop
```
6. Clone this github repository
```
git clone https://github.com/Bart-Steemans/bactoscoop.git
```
7. Set bactoscoop as your current directory and install the remaining requirements
```
   cd /path/to/bactoscoop/
```
```
   pip install -r requirements.txt
```


# Tutorial

An example notebook can be found here [Notebook](https://github.com/Bart-Steemans/bactoscoop/blob/main/bactoscoop_demo.ipynb)

# Common issues
If you encounter following error [ImportError: cannot import name 'ifft' from 'scipy'](https://github.com/kevinjohncutler/omnipose/issues/78) 
Fix this by replacing 
```
from scipy import fft, ifft
```
with
```
from scipy.fft import fft, ifft
```

