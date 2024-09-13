# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:08:02 2023

@author: Bart Steemans. Govers Lab.
"""

from cellpose_omni import models, core, io
from cellpose_omni import models
from cellpose_omni.models import MODEL_NAMES


class Omnipose:

    def __init__(self, imgs, paths):

        self.imgs = imgs
        self.files = paths
        self.use_GPU = core.use_gpu()
        self.masks = None
        self.flows = None
        self.model = None
        print(f">>> GPU activated? {self.use_GPU}")

    def load_models(self, model_name):
        """
        Load the desired omnipose model.

        """
        if self.model is None:
            self.model = models.CellposeModel(gpu=self.use_GPU, model_type=model_name)

    def process(self, n=[0], mask_thresh=1, minsize=300):
        """
        Segment the raw images and get their masks.

        """
        chans = [0, 0]  # th is means segment based on first channel, no second channel
        mask_threshold = mask_thresh  # negative = mask becomes larger, positive = mask becomes thinner
        verbose = 0  # turn on if you want to see more output
        use_gpu = self.use_GPU  # defined above
        transparency = True  # transparency in flow output
        rescale = (
            None  # give this a number if you need to upscale or downscale your images
        )
        omni = True  # we can turn off Omnipose mask reconstruction, not advised
        flow_threshold = 0.0  # default is .4, but only needed if there are spurious masks to clean up; slows down output
        resample = (
            True  # whether or not to run dynamics on rescaled grid or original grid
        )
        batch_size = 1
        niter = None
        # diamter: too small -> oversegmentation, too large -> undersegmentation
        # if diameter = None, the diam will be estimated (in images from test-set )
        self.masks, self.flows, styles = self.model.eval(
            [self.imgs[i] for i in n],
            niter=niter,
            channels=chans,
            rescale=rescale,
            mask_threshold=mask_threshold,
            transparency=transparency,
            flow_threshold=flow_threshold,
            omni=omni,
            resample=resample,
            cluster=True,
            verbose=verbose,
            diameter=None,
            min_size=minsize,
            batch_size=batch_size,
        )

    def save_masks(self):
        """
        Save the segmentation masks.

        """
        io.save_masks(
            self.imgs,
            self.masks,
            self.flows,
            self.files,
            png=False,
            tif=True,  # whether to use PNG or TIF format
            suffix="",  # suffix to add to files if needed
            save_flows=False,  # saves both RGB depiction as *_flows.png and the raw components as *_dP.tif
            save_outlines=False,  # save outline images
            dir_above=0,  # save output in the image directory or in the directory above (at the level of the image directory)
            in_folders=True,  # save output in folders (recommended)
            save_txt=False,  # txt file for outlines in imageJ
            save_ncolor=False,
        )  # save ncolor version of masks for visualization and editing

    def compiled_process(self, n, mask_thresh, minsize):
        """
        Sequentially process and save the masks in one method.
        """
        self.process(n, mask_thresh, minsize)
        self.save_masks()
