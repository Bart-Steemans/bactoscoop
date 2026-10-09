# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:47:19 2023

@author: Bart Steemans. Govers Lab.
"""

import os
import pickle
import cv2 as cv
import numpy as np
import tifffile
from . import utilities as u
from .logging_utils import get_bactoscoop_logger, tqdm_if_verbose
from .image import Image
import pandas as pd
from itertools import combinations

import logging, sys, datetime, traceback
import gc
import tempfile


bactoscoop_logger = get_bactoscoop_logger()
# # Change following to DEBUG to see more information
# bactoscoop_logger.setLevel(logging.INFO)


class ImageCollection:
    """ """

    def __init__(
        self, image_folder_path=None, px=0.065, error_policy="continue",
        log_cell_errors=False,
    ):
        """Create a collection.

        ``error_policy="continue"`` records item-level errors and proceeds with
        independent fields. ``error_policy="raise"`` raises on the first one.
        Required-stage failures still stop dependent work in either mode.
        Cell feature failures are recorded separately and do not make a run
        partial in continue mode. Set ``log_cell_errors=True`` to print each one.
        """
        if error_policy not in {"continue", "raise"}:
            raise ValueError("error_policy must be either 'continue' or 'raise'.")
        self.error_policy = error_policy
        self.log_cell_errors = log_cell_errors
        self.processing_errors = []
        self.cell_errors = []
        self.completed_stages = set()

        self.px = px  # microns per pixel
        self.images = None
        self.masks = None
        self.channel_images = {}
        self.channel_image_filenames = {}
        self.channel_images_by_field = {}
        self.inverted_image = None
        self.channel_list = None
        self.image_filenames = None
        self.mask_filenames = None
        self.phase_channel = ""
        self.segmentation_failed = False

        self.image_folder_path = image_folder_path  # Path to the folder to be processed
        self.name = (
            os.path.basename(os.path.normpath(self.image_folder_path))
            if self.image_folder_path is not None
            else None
        )

        self.image_objects = []  # Array that will be populated with objects

        self.mesh_df_collection = pd.DataFrame()

        self.feature_dataframes = {}
        self.merged_features = None
        self.mesh_prefilter_stats = pd.DataFrame()

    def _require_image_folder_path(self):
        if self.image_folder_path is None:
            raise ValueError("image_folder_path must be set before loading image data.")

    def _require_loaded_images(self):
        if not self.image_objects:
            raise ValueError(
                "No image objects are available. Run create_image_objects() or load mesh data first."
            )

    def _record_processing_error(
        self, stage, error, image_name=None, channel=None, cell_id=None
    ):
        record = {
            "stage": stage,
            "image_name": image_name,
            "channel": channel,
            "cell_id": cell_id,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        self.processing_errors.append(record)
        bactoscoop_logger.error(
            "Processing error at %s (image=%s, channel=%s, cell_id=%s): %s",
            stage,
            image_name,
            channel,
            cell_id,
            error,
        )
        if self.error_policy == "raise":
            raise error
        return record

    def _record_cell_error(
        self, stage, error, image_name, channel, cell_id, error_type=None
    ):
        record = {
            "stage": stage,
            "image_name": image_name,
            "channel": channel,
            "cell_id": cell_id,
            "error_type": error_type or type(error).__name__,
            "error": str(error),
        }
        self.cell_errors.append(record)
        if self.log_cell_errors:
            bactoscoop_logger.warning(
                "Cell processing error at %s (image=%s, channel=%s, cell_id=%s): %s",
                stage, image_name, channel, cell_id, error,
            )
        if self.error_policy == "raise":
            raise error
        return record

    def processing_summary(self):
        """Return attempted-stage status and retained image/cell error records."""
        if self.processing_errors:
            status = "completed_with_errors"
        elif self.completed_stages:
            status = "completed"
        else:
            status = "not_started"
        return {
            "status": status,
            "completed_stages": sorted(self.completed_stages),
            "error_count": len(self.processing_errors),
            "errors": list(self.processing_errors),
            "cell_error_count": len(self.cell_errors),
            "cell_errors": list(self.cell_errors),
        }

    def _record_stage_success(self, stage):
        self.completed_stages.add(stage)

    def _validate_phase_image_state(self):
        if self.images is None or self.image_filenames is None:
            raise ValueError(
                "Phase images are not loaded. Run load_phase_images() before creating image objects."
            )
        if self.masks is None:
            raise ValueError(
                "Masks are not loaded. Run load_masks() before creating image objects."
            )
        if len(self.image_filenames) != len(self.images):
            raise ValueError(
                f"Loaded {len(self.image_filenames)} image names but {len(self.images)} phase images."
            )
        if self.mask_filenames is None or len(self.mask_filenames) != len(self.masks):
            raise ValueError("Mask arrays and mask filenames are not aligned.")

        phase_by_key = {}
        for index, filename in enumerate(self.image_filenames):
            key = u.image_field_key(filename, suffix=self.phase_channel)
            if key in phase_by_key:
                self._record_processing_error(
                    "phase_mask_matching",
                    ValueError(f"Duplicate phase field identity '{key}'."),
                    image_name=filename,
                )
                continue
            phase_by_key[key] = (index, filename, self.images[index])

        masks_by_key = {}
        for index, filename in enumerate(self.mask_filenames):
            key = u.image_field_key(
                filename, suffix=self.phase_channel, is_mask=True
            )
            if key in masks_by_key:
                self._record_processing_error(
                    "phase_mask_matching",
                    ValueError(f"Duplicate mask field identity '{key}'."),
                    image_name=filename,
                )
                continue
            masks_by_key[key] = (filename, self.masks[index])

        matched = []
        for key, (frame, image_name, image) in phase_by_key.items():
            mask_entry = masks_by_key.get(key)
            if mask_entry is None:
                self._record_processing_error(
                    "phase_mask_matching",
                    ValueError(f"No mask matches phase field '{image_name}'."),
                    image_name=image_name,
                )
                continue
            mask_name, mask = mask_entry
            if image.ndim != 2 or mask.ndim != 2:
                self._record_processing_error(
                    "phase_mask_matching",
                    ValueError(
                        f"Phase and mask images must be 2D; got phase shape "
                        f"{image.shape} and mask '{mask_name}' shape {mask.shape}."
                    ),
                    image_name=image_name,
                )
                continue
            if image.shape != mask.shape:
                self._record_processing_error(
                    "phase_mask_matching",
                    ValueError(
                        f"Phase shape {image.shape} does not match mask "
                        f"'{mask_name}' shape {mask.shape}."
                    ),
                    image_name=image_name,
                )
                continue
            matched.append((frame, image_name, image, mask))

        for key, (mask_name, _) in masks_by_key.items():
            if key not in phase_by_key:
                self._record_processing_error(
                    "phase_mask_matching",
                    ValueError(f"No phase image matches mask '{mask_name}'."),
                    image_name=mask_name,
                )
        if not matched:
            raise ValueError("No phase images have matching masks and dimensions.")
        return matched
    def _normalize_channel_list(self, channel_list, argument_name="channel_list"):
        if channel_list is None:
            raise ValueError(
                f"No {argument_name} provided. It should be of format ['channelname1', 'channelname2']."
            )
        if isinstance(channel_list, str):
            raise TypeError(
                f"{argument_name} must be an iterable of channel names, not a single string."
            )
        normalized_channels = list(channel_list)
        if not normalized_channels:
            raise ValueError(f"{argument_name} must contain at least one channel name.")
        return normalized_channels

    def _iter_processing_targets(self, object_list, phase_channel):
        if isinstance(object_list, Image):
            return [object_list]
        if object_list is None:
            self.create_image_objects(phase_channel=phase_channel)
            return self.image_objects
        if isinstance(object_list, (list, tuple)):
            return list(object_list)
        raise TypeError(
            "object_list must be None, a single Image instance, or a list/tuple of Image instances."
        )

    def _get_mesh_dataframe_lookup(self):
        """Build an image-name lookup from the current mesh dataframe.

        Do not retain this lookup across calls: callers can replace or mutate
        ``mesh_df_collection``, and a cached lookup could then return stale
        meshes for newly loaded data.
        """
        if not isinstance(self.mesh_df_collection, pd.DataFrame) or self.mesh_df_collection.empty:
            return None
        return {
            image_name: group.reset_index(drop=True)
            for image_name, group in self.mesh_df_collection.groupby(
                "image_name", sort=False
            )
        }


    def load_masks(self):
        """
        Read masks from the ``/masks`` subfolder.

        Mutates ``self.masks`` and ``self.mask_filenames``.

        """
        self._require_image_folder_path()
        self.masks, self.mask_filenames = u.read_tiff_folder(
            self.image_folder_path + "/masks"
        )
        self._record_stage_success("load_masks")

    def load_phase_images(self, phase_channel=""):
        """
        This method loads phase contrast images from the specified image folder path, based on the characters before the file-extension,
        and stores them in the 'self.images' array.

        Parameters
        ----------
        phase_channel : str, optional
            Characters before the file-extension (e.g., "c1", "phase") to look for when using this function. (default is an empty string)

        Mutates ``self.images``, ``self.image_filenames`` and ``self.paths``.

        """
        self._require_image_folder_path()
        self.phase_channel = phase_channel or ""
        try:
            self.images, self.image_filenames, self.paths = u.read_tiff_folder(
                self.image_folder_path, phase_channel, include_paths=True
            )
        except Exception as e:
            raise type(e)(
                f"Failed to load phase contrast images from '{self.image_folder_path}' "
                f"with phase_channel='{phase_channel}': {e}"
            ) from e
        self._record_stage_success("load_phase_images")

    def load_channel_images(self, channel_list):
        """
        This method loads channel images from the image folder path based on the specified characters before the file-extension and stores them in
        the 'self.channel_images' dictionary, where the channel name is used as the key.

        Parameters
        ----------
        channel_list : List of str
            Characters before the extension (e.g., ["c2", "c3"]) to look for when using this function.

        Mutates ``self.channel_images``.

        """
        self._require_image_folder_path()
        self.channel_list = self._normalize_channel_list(channel_list)
        for channel in self.channel_list:
            # A failed read must not leave arrays from an earlier folder/load in place.
            self.channel_images.pop(channel, None)
            self.channel_image_filenames.pop(channel, None)
            self.channel_images_by_field.pop(channel, None)
            try:
                images, filenames = u.read_tiff_folder(
                    self.image_folder_path, suffix=channel
                )
            except Exception as e:
                self._record_processing_error(
                    "load_channel_images", e, channel=channel
                )
                continue

            self.channel_images[channel] = list(images)
            self.channel_image_filenames[channel] = list(filenames)
            keyed_images = {}
            for filename, image in zip(filenames, images):
                key = u.image_field_key(filename, suffix=channel)
                if key in keyed_images:
                    self._record_processing_error(
                        "load_channel_images",
                        ValueError(f"Duplicate channel field identity '{key}'."),
                        image_name=filename,
                        channel=channel,
                    )
                    continue
                keyed_images[key] = image
            self.channel_images_by_field[channel] = keyed_images
        self._record_stage_success("load_channel_images")

    def create_image_objects(self, phase_channel=""):
        """
        Create ``Image`` objects from the loaded masks and phase images.

        Invalid collection-level setup raises immediately. Failures while creating
        a single image object are logged and skipped so the rest of the batch can continue.
        """
        if self.segmentation_failed:
            raise RuntimeError(
                "Segmentation failed in this collection; refusing to load masks "
                "from disk because they may belong to an earlier run."
            )
        if self.masks is None:
            self.load_masks()
        if self.images is None or self.image_filenames is None:
            self.load_phase_images(phase_channel)

        matched_images = self._validate_phase_image_state()

        # Feature tables and merged output belong to the previous image/cell set.
        self.feature_dataframes = {}
        self.merged_features = None
        self.image_objects = []
        mesh_lookup = self._get_mesh_dataframe_lookup()

        for i, image_name, image, mask in matched_images:
            try:
                mesh_df = mesh_lookup.get(image_name) if mesh_lookup is not None else None

                img_obj = self.create_image_object(
                    image, image_name, i, mask, mesh_df, self.px
                )
                self.image_objects.append(img_obj)
            except Exception as e:
                bactoscoop_logger.debug(traceback.format_exc())
                self._record_processing_error(
                    "create_image_objects", e, image_name=image_name
                )
        self._record_stage_success("create_image_objects")

    def add_channels(self, img_objects, channel_list, load_data=True):
        """
        This method loads channel images based on the provided channel list and adds them to the individual image objects.

        Parameters
        ----------
        img_objects : List
            List of image objects.
        channel_list : List
            List of channel names.

        """
        channel_list = self._normalize_channel_list(channel_list)
        if img_objects is None or len(img_objects) == 0:
            raise ValueError("img_objects must contain at least one image object.")

        # Load requested channels while keeping field identity beside each array.
        if load_data:
            self.load_channel_images(channel_list)

        expected_fields = {
            u.image_field_key(obj.image_name, suffix=self.phase_channel)
            for obj in img_objects
        }
        for channel in channel_list:
            for extra_field in self.channel_images_by_field.get(channel, {}):
                if extra_field not in expected_fields:
                    self._record_processing_error(
                        "match_channel_image",
                        ValueError(
                            f"No phase image object matches {channel} field '{extra_field}'."
                        ),
                        image_name=extra_field,
                        channel=channel,
                    )

        for img_obj in img_objects:
            channel_single_image_dict = dict(getattr(img_obj, "channels", None) or {})
            field_key = u.image_field_key(
                img_obj.image_name, suffix=self.phase_channel
            )
            for channel in channel_list:
                # Never retain an earlier image or derived interpolation for a
                # requested channel if its current file is missing or invalid.
                channel_single_image_dict.pop(channel, None)
                for cache_name in ("bg_channels", "chann_interp2d"):
                    cache = getattr(img_obj, cache_name, None)
                    if isinstance(cache, dict):
                        cache.pop(channel, None)
                channel_fields = self.channel_images_by_field.get(channel, {})
                channel_image = channel_fields.get(field_key)
                if channel_image is None:
                    self._record_processing_error(
                        "match_channel_image",
                        ValueError(
                            f"No {channel} image matches field '{img_obj.image_name}'."
                        ),
                        image_name=img_obj.image_name,
                        channel=channel,
                    )
                    continue
                if channel_image.ndim != 2 or img_obj.image.ndim != 2:
                    self._record_processing_error(
                        "match_channel_image",
                        ValueError(
                            f"{channel} and phase images must be 2D; got channel "
                            f"shape {channel_image.shape} and phase shape {img_obj.image.shape}."
                        ),
                        image_name=img_obj.image_name,
                        channel=channel,
                    )
                    continue
                if channel_image.shape != img_obj.image.shape:
                    self._record_processing_error(
                        "match_channel_image",
                        ValueError(
                            f"{channel} image shape {channel_image.shape} does not "
                            f"match phase shape {img_obj.image.shape}."
                        ),
                        image_name=img_obj.image_name,
                        channel=channel,
                    )
                    continue
                channel_single_image_dict[channel] = channel_image
            img_obj.channels = channel_single_image_dict or None
        self._record_stage_success("add_channels")

    def get_mesh_dataframe(self, image_name):
        """
        This method returns the mesh dataframe associated with the specified image name.

        Parameters
        ----------
        image_name : str
            The name or identifier of the image.

        """

        mesh_lookup = self._get_mesh_dataframe_lookup()
        if mesh_lookup is None:
            return None
        return mesh_lookup.get(image_name)

    def create_image_object(self, image, image_name, index, mask, mesh_df, px):
        """
        This method creates a single Image object using the provided image, image name, index, mask, mesh dataframe, and pixel size (px).
        If mesh data is present, it also creates associated Cell objects.

        Parameters
        ----------
        image : phase contrast image
            The phase contrast image for the Image object.
        image_name : str
            The name or identifier of the image.
        index : int
            The index or position of the image.
        mask : mask or None
            The mask or region of interest associated with the image.
        mesh_df : pd.DataFrame or None
            The mesh dataframe associated with the image, if available, or None.
        px : float
        The pixel size in micrometers.
        """

        img_obj = Image(image, image_name, index, mask, mesh_df, px=px)

        if mesh_df is not None:
            img_obj.create_cell_object(verbose=False)
        return img_obj

    # Methods for batch processing of images ----------------------------------------------
    def segment_images(self, mask_thresh, minsize, n, model_name="bact_phase_omni"):
        self.segmentation_failed = False
        self.masks = None
        self.mask_filenames = None
        try:
            from .omni import Omnipose
            import torch

            omni = Omnipose(self.images, self.paths)
            omni.load_models(model_name)
    
            omni.compiled_process(n, mask_thresh, minsize)
            torch.cuda.empty_cache()
    
            del omni
            gc.collect()
            self._record_stage_success("segment_images")
            return True
        except Exception as e:
            self.segmentation_failed = True
            bactoscoop_logger.debug(traceback.format_exc())
            self._record_processing_error("segment_images", e)
            return False

    def batch_detect_objects(
        self,
        channels=None,
        reset_channels=True,
        align=False,
        smoothing=0.1,
        log_sigma=3,
        kernel_width=4,
        min_overlap_ratio=0.01,
        max_external_ratio=0.1,
    ):
        """
        Detect objects within the contour of the cell.

        Parameters
        ----------
        channels : List of strings, optional
            The channel names of the channels to perform object detection on.

        log_sigma : int, optional
            The sigma parameter of the Laplacian of Gaussian filter used in object detection. (default is 3)

        kernel_width : int, optional
            The kernel width of the kernel used for dilating the object masks.
            Decreasing results in smaller masks, while increasing results in larger masks. (default is 4)

        min_overlap_ratio : float, optional
            Minimum overlap between the object and cell required for the object to be kept. (default is 0.01)

        max_external_ratio : float, optional
            The maximum ratio an object is allowed to lie outside the cell contour. (default is 0.1)

        Returns
        -------
        pd.DataFrame
            A DataFrame containing cell_id, object_contours, and frame information.

        """
        self._require_loaded_images()
        channels = self._normalize_channel_list(channels, argument_name="channels")

        if reset_channels:
            self.channel_images = {}
            self.channel_image_filenames = {}
            self.channel_images_by_field = {}
            for image in self.image_objects:
                image.channels = None
                for cache_name in ("bg_channels", "chann_interp2d"):
                    cache = getattr(image, cache_name, None)
                    if isinstance(cache, dict):
                        cache.clear()

        channels_to_load = [
            channel for channel in channels
            if channel not in self.channel_images_by_field
        ]
        if channels_to_load:
            self.load_channel_images(channels_to_load)
        self.add_channels(self.image_objects, channels, load_data=False)

        bactoscoop_logger.info("Detecting objects within cell ...")

        dfs = []
        for image in tqdm_if_verbose(self.image_objects):
            for cell in getattr(image, "cells", []):
                for channel in channels:
                    cell.object_meshdata.pop(channel, None)
            if image.channels is None or any(
                channel not in image.channels for channel in channels
            ):
                # The missing field has already been recorded by add_channels.
                continue
            try:
                detection_df = image.object_detection(
                    channels,
                    smoothing,
                    align,
                    log_sigma,
                    kernel_width,
                    min_overlap_ratio,
                    max_external_ratio,
                )
            except Exception as e:
                bactoscoop_logger.debug(traceback.format_exc())
                self._record_processing_error(
                    "object_detection", e, image_name=image.image_name
                )
                continue
            dfs.append(detection_df)
            cell_errors = getattr(image, "object_detection_errors", [])
            for cell_error in cell_errors:
                self._record_cell_error(
                    "cell_object_detection",
                    RuntimeError(cell_error["error"]),
                    image_name=image.image_name,
                    channel=cell_error.get("channel"),
                    cell_id=cell_error.get("cell_id"),
                    error_type=cell_error.get("error_type"),
                )
            cell_count = len(getattr(image, "cells", []))
            failed_attempts = sum(
                not error.get("recovered", False) for error in cell_errors
            )
            if cell_count and failed_attempts >= cell_count * len(channels):
                self._record_processing_error(
                    "object_detection",
                    ValueError(
                        f"All cell object-detection attempts failed for '{image.image_name}'."
                    ),
                    image_name=image.image_name,
                )

        self.object_detection_df = (
            pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()
        )

        if not self.object_detection_df.empty and "channel" in self.object_detection_df:
            bactoscoop_logger.info(self.object_detection_df["channel"].value_counts())
        else:
            bactoscoop_logger.info("No objects were detected.")

        self._record_stage_success("batch_detect_objects")
        return self.object_detection_df

    def batch_process_mesh(
        self,
        pkl_name=None,
        object_list=None,
        save_data=True,
        phase_channel="",
        join_thresh=4,
        split_thresh=0.35,
        CD_width=False,
        smoothing=0.1,
        neighbor_filter_max_neighbors=None,
        neighbor_filter_connectivity=1,
        max_daughter_cell_mesh_rows=800,
    ):
        """
        Process cellular meshes from segmented masks and store the data in Cell objects.

        After segmentation, this method can process the resulting masks and create cellular meshes from them.
        It first joins cells with a pole-pole distance (px) below the 'join_thresh'.
        Subsequently, meshes of all cells are created and cells can be split again if the constriction degree exceeds the 'split_thresh'.
        The data is then stored in Cell objects and can be exported in a pickle file.

        Parameters
        ----------
        pkl_name : str, optional
            Name of the pickle file to save the mesh data. If None, the file will be saved as 'folder_name_meshdata.pkl'.

        object_list : List, optional
            Specify a list of Image objects to be processed. If 'object_list' is None, all Image instances in 'image_objects' are processed.

        phase_channel : str, optional
            Characters before the file-extension (.tif, .tiff) that the algorithm looks for when using this function (e.g., "c1", "phase").

        join_thresh : int, optional
            The pole-pole distance threshold for joining cells. Cells with a distance below this threshold will have their masks joined. (default is 4)

        split_thresh : float, optional
            The constriction degree threshold for splitting cells. Cells with a constriction degree exceeding this threshold will be split. (default is 0.35)

        max_daughter_cell_mesh_rows : int, optional
            Maximum rebuilt daughter mesh rows (default 800, minimum 4).
            Oversized resampling is rejected before allocation; both final
            daughters must have 4..limit rows. This is separate from the feature
            extraction max_mesh_size parameter and does not limit unsplit parents.

        """
        max_daughter_cell_mesh_rows = u._validate_max_daughter_cell_mesh_rows(
            max_daughter_cell_mesh_rows
        )
        self.mesh_df_collection = pd.DataFrame()
        self.mesh_prefilter_stats = pd.DataFrame()
        object_list = self._iter_processing_targets(object_list, phase_channel)
        mesh_frames = []
        prefilter_stats = []

        for img_obj in object_list:
            bactoscoop_logger.info(f"PROCESSING IMAGE: {img_obj.image_name}")
            try:
                img_obj.join_split_pipeline(
                    join_thresh,
                    split_thresh,
                    CD_width,
                    smoothing,
                    neighbor_filter_max_neighbors=neighbor_filter_max_neighbors,
                    neighbor_filter_connectivity=neighbor_filter_connectivity,
                    max_daughter_cell_mesh_rows=max_daughter_cell_mesh_rows,
                )
                mesh_df = getattr(img_obj, "processed_mesh_dataframe", None)
                if mesh_df is None:
                    raise ValueError(
                        f"Image '{img_obj.image_name}' did not produce a processed mesh dataframe."
                    )
                mesh_frames.append(mesh_df)
                if getattr(img_obj, "prefilter_summary", None) is not None:
                    prefilter_stats.append(
                        {
                            "image_name": img_obj.image_name,
                            **img_obj.prefilter_summary,
                        }
                    )
                img_obj.create_cell_object(verbose=False)
            except Exception as e:
                bactoscoop_logger.debug(traceback.format_exc())
                self._record_processing_error(
                    "mesh_processing", e, image_name=img_obj.image_name
                )

        self.mesh_df_collection = (
            pd.concat(mesh_frames, ignore_index=True) if mesh_frames else pd.DataFrame()
        )
        self.mesh_prefilter_stats = (
            pd.DataFrame(prefilter_stats) if prefilter_stats else pd.DataFrame()
        )

        if save_data:
            if pkl_name is not None:
                self._to_pickle(pkl_name, self.mesh_df_collection)

            self._to_pickle(f"{self.name}_meshdata.pkl", self.mesh_df_collection)

        self._record_stage_success("batch_process_mesh")


    def batch_load_mesh(self, pkl_name, pkl_path=None, phase_channel=""):
        """
        Load mesh data from a pickle-file and create image objects from the data.

        Parameters
        ----------
        pkl_name : str
            Name of the pickle file where the mesh data is stored (e.g., 'Ecoli_meshdata.pkl').

        pkl_path : str, optional
            Path to the file where the mesh data is stored. If not provided, the method will look in the image folder path.

        phase_channel : str, optional
            Characters before the file-extension (.tif, .tiff) that the algorithm looks for when using this function (e.g., "c1", "phase").

        """
        self.mesh_df_collection = None
        if pkl_path is None:
            pkl_path = self.image_folder_path

        full_path = os.path.join(pkl_path, pkl_name)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Pickle file not found: {full_path}")

        if full_path.lower().endswith(".parquet"):
            mesh_df = pd.read_parquet(full_path, engine="pyarrow")
            for column in ("contour", "mesh", "midline"):
                if column not in mesh_df:
                    raise ValueError(f"Mesh Parquet file is missing '{column}'.")
                mesh_df[column] = mesh_df[column].map(
                    lambda value: np.asarray(value.tolist(), dtype=float)
                )
            self.mesh_df_collection = mesh_df
        else:
            with open(full_path, "rb") as f:
                self.mesh_df_collection = pickle.load(f)

        bactoscoop_logger.info("Meshes loaded from file: %s", pkl_name)
        self.create_image_objects(phase_channel=phase_channel)

    def batch_mask2mesh(
        self,
        pkl_name=None,
        object_list=None,
        save_data=True,
        phase_channel="",
        smoothing=0.1,
    ):
        """
        Create meshes from masks without the joining or splitting as in the batch_process_mesh method.

        Parameters
        ----------
        pkl_name : str, optional
            Name of the pickle file to save the mesh data. If None, the file will be saved as 'folder_name_meshdata.pkl'.
        object_list : List, optional
            Specify a list of Image objects to be processed. If 'object_list' is None, all Image instances in 'image_objects' are processed.
        phase_channel : str, optional
            Characters before the file-extension (.tif, .tiff) that the algorithm looks for when using this function (e.g., "c1", "phase").
        """
        self.mesh_df_collection = pd.DataFrame()
        mesh_frames = []

        if isinstance(object_list, Image):
            print("\n")
            bactoscoop_logger.info(f"PROCESSING IMAGE: {object_list.image_name}")

            object_list.mask2mesh(smoothing)
            mesh_frames.append(getattr(object_list, "mesh_dataframe"))
            object_list.create_cell_object(verbose=False)
        else:

            if object_list is None:
                self.create_image_objects(phase_channel=phase_channel)
                object_list = self.image_objects

            for img_obj in object_list:

                bactoscoop_logger.info(f"PROCESSING IMAGE: {img_obj.image_name}")

                img_obj.mask2mesh()
                mesh_frames.append(getattr(img_obj, "mesh_dataframe"))

                img_obj.create_cell_object(verbose=False)

        self.mesh_df_collection = (
            pd.concat(mesh_frames, ignore_index=True) if mesh_frames else pd.DataFrame()
        )

        if save_data:
            if pkl_name is not None:
                self._to_pickle(pkl_name, self.mesh_df_collection)

            self._to_pickle(
                "{}_meshdata.pkl".format(self.name), self.mesh_df_collection
            )

    def batch_shift_correction(
        self,
        shifted_channel,
        reset_channels=False,
        log_sigma=1.5,
        kernel_width=3,
        min_overlap_ratio=0.1,
        max_external_ratio=1,
        phase_log_sigma=0.5,
        phase_closing_level=2,
        signal_closing_level=12,
        max_shift_correction=15,
    ):
        self._require_loaded_images()
        shifted_channel = self._normalize_channel_list(
            shifted_channel, argument_name="shifted_channel"
        )

        if reset_channels:
            self.channel_images = {}
            self.channel_image_filenames = {}
            self.channel_images_by_field = {}
            for image in self.image_objects:
                image.channels = None
                for cache_name in ("bg_channels", "chann_interp2d"):
                    cache = getattr(image, cache_name, None)
                    if isinstance(cache, dict):
                        cache.clear()

        channels_to_load = [
            channel for channel in shifted_channel
            if channel not in self.channel_images_by_field
        ]
        if channels_to_load:
            self.load_channel_images(channels_to_load)
        self.add_channels(self.image_objects, shifted_channel, load_data=False)

        success_percentage_list = []
        bactoscoop_logger.info("Aligning contour, mesh and midline to channel image ...")
        for image_obj in tqdm_if_verbose(self.image_objects):
            if image_obj.channels is None or any(
                channel not in image_obj.channels for channel in shifted_channel
            ):
                # Missing channels were recorded by add_channels.
                continue
            try:
                success_percentage = image_obj.shift_correction(
                    shifted_channel,
                    log_sigma,
                    kernel_width,
                    min_overlap_ratio,
                    max_external_ratio,
                    phase_log_sigma,
                    phase_closing_level,
                    signal_closing_level,
                    max_shift_correction,
                )
            except Exception as e:
                bactoscoop_logger.debug(traceback.format_exc())
                self._record_processing_error(
                    "shift_correction", e, image_name=image_obj.image_name
                )
                continue
            success_percentage_list.append(success_percentage)
            cell_errors = getattr(image_obj, "shift_errors", [])
            for cell_error in cell_errors:
                self._record_cell_error(
                    "cell_shift_correction",
                    RuntimeError(cell_error["error"]),
                    image_name=image_obj.image_name,
                    channel=cell_error.get("channel"),
                    cell_id=cell_error.get("cell_id"),
                    error_type=cell_error.get("error_type"),
                )
            cell_count = len(getattr(image_obj, "cells", []))
            if cell_count and len(cell_errors) >= cell_count * len(shifted_channel):
                self._record_processing_error(
                    "shift_correction",
                    ValueError(
                        f"All cell shift-correction attempts failed for '{image_obj.image_name}'."
                    ),
                    image_name=image_obj.image_name,
                )
        if not success_percentage_list:
            bactoscoop_logger.warning("No shift-correction results were produced.")
            return 0.0

        average_success_percentage = sum(success_percentage_list) / len(
            success_percentage_list
        )

        bactoscoop_logger.info(f"Shift succes rate: {average_success_percentage}")
        return average_success_percentage

    def batch_calculate_features(
        self,
        channel_method_pairs,
        all_data=True,
        reset=True,
        use_shifted_contours=False,
        shift_signal=False,
        max_mesh_size=600,
        retain_contour_on_object_mesh_failure_channels=None,
    ):
        """
        Calculate features of all image objects for specified channels and methods.

        This method calculates features for all image objects based on the specified channel-method pairs. It takes a list of tuples, where each tuple contains channel names and the feature calculation method. The resulting feature dataframes are stored in a dictionary.

        Parameters
        ----------
        channel_method_pairs : List[Tuple[List[str], str]]
            A list of tuples where each tuple contains the channel names and feature calculation method.
            Example of channel_method_pairs: [(['c1'], 'profiling'), (['c2', 'c3'], 'objects'), ([None], 'svm')].
            With the example pairs, this method will analyze c1 using the profiling method,
            then c2 and c3 using the objects method, and finally the phase contrast channel using the svm method.

        add_profiling_data : bool, optional
            Whether to include profiling data in the feature calculation. (default is True)

        reset : bool, optional
            If True, the dictionary storing the dataframes will be reset. If False, dataframes are added to the existing dictionary. (default is True)
        retain_contour_on_object_mesh_failure_channels : list[str] or None, optional
            Channels whose objects features retain contour-based measurements
            when any detected object mesh is invalid. Other channels omit that
            cell's objects feature row. None is equivalent to an empty list.
            This option does not affect other feature methods.

        Returns
        -------
        Dict
            A dictionary containing dataframes with calculated features.

        """
        self._require_loaded_images()
        if channel_method_pairs is None:
            raise ValueError("channel_method_pairs must be provided.")
        if retain_contour_on_object_mesh_failure_channels is None:
            retained_channels = frozenset()
        elif isinstance(retain_contour_on_object_mesh_failure_channels, str):
            raise TypeError(
                "retain_contour_on_object_mesh_failure_channels must be a list "
                "of channel names, not a string."
            )
        else:
            retained_channels = frozenset(
                retain_contour_on_object_mesh_failure_channels
            )
        if not all(isinstance(channel, str) for channel in retained_channels):
            raise TypeError(
                "retain_contour_on_object_mesh_failure_channels must contain "
                "only channel names."
            )

        self.merged_features = None
        if reset:
            self.feature_dataframes = {}

        for channels, method in channel_method_pairs:
            normalized_channels = (
                [None]
                if channels == [None]
                else self._normalize_channel_list(channels, argument_name="channels")
            )
            for channel in normalized_channels:
                key = f"{method}_{channel}_features"
                self.feature_dataframes.pop(key, None)
                bactoscoop_logger.info(
                    f"PROCESSING CHANNEL: {channel}, method: {method}"
                )
                feature_dfs = []

                for img_obj in tqdm_if_verbose(self.image_objects):
                    if channel is not None and (
                        img_obj.channels is None or channel not in img_obj.channels
                    ):
                        self._record_processing_error(
                            "image_feature_calculation",
                            ValueError(
                                f"Channel {channel} is not loaded for image '{img_obj.image_name}'."
                            ),
                            image_name=img_obj.image_name,
                            channel=channel,
                        )
                        continue
                    try:
                        result_df = img_obj.calculate_features(
                            method,
                            channel,
                            all_data,
                            use_shifted_contours,
                            shift_signal,
                            max_mesh_size,
                            retained_channels,
                        )
                    except Exception as e:
                        bactoscoop_logger.debug(traceback.format_exc())
                        self._record_processing_error(
                            "image_feature_calculation",
                            e,
                            image_name=img_obj.image_name,
                            channel=channel,
                        )
                        continue
                    if result_df is not None:
                        feature_dfs.append(result_df)
                    for feature_error in getattr(img_obj, "feature_errors", []):
                        self._record_cell_error(
                            "cell_feature_calculation",
                            RuntimeError(feature_error["error"]),
                            image_name=img_obj.image_name,
                            channel=channel,
                            cell_id=feature_error.get("cell_id"),
                            error_type=feature_error.get("error_type"),
                        )

                if feature_dfs:
                    concatenated_df = pd.concat(feature_dfs, axis=0)
                    concatenated_df.reset_index(drop=True, inplace=True)
                    self.feature_dataframes[key] = concatenated_df
                    has_rows = not concatenated_df.empty
                else:
                    has_rows = False
                if not has_rows:
                    self._record_processing_error(
                        "batch_calculate_features",
                        ValueError(
                            f"No feature rows were produced for channel {channel}, method {method}."
                        ),
                        channel=channel,
                    )

        self._record_stage_success("batch_calculate_features")
        return self.feature_dataframes

    def batch_calculate_signal_correlation_features(
        self, df, channels, feature_method_tuples
    ):
        from .signalcorrelation import SignalCorrelation

        # Generate all unique pairs of channels
        channel_pairs = combinations(channels, 2)

        for channel1, channel2 in channel_pairs:
            for features, method_names in feature_method_tuples:
                bactoscoop_logger.info(
                    f"Processing channel {channel1} vs {channel2} and features {features} from {self.image_folder_path}:"
                )
                for feature in tqdm_if_verbose(features):
                    prepared_feature = SignalCorrelation.prepare_feature_pair(
                        df, channel1, channel2, feature
                    )

                    for method_name in method_names:
                        sc = SignalCorrelation(
                            df,
                            channel1,
                            channel2,
                            feature,
                            method_name,
                            prepared_feature=prepared_feature,
                        )
                        sc.calculate()
                        del sc
        return df

    def _apply_curation_labels(self):
        """Apply curation labels by stable image identity, with frame fallback."""
        image_objects_by_name = {}
        image_objects_by_frame = {}
        for image in self.image_objects:
            image_objects_by_name.setdefault(image.image_name, []).append(image)
            image_objects_by_frame.setdefault(image.frame, []).append(image)

        for _, row in self.curated_df.iterrows():
            image_name = row.get("image_name")
            cell_id = row["cell_id"]
            label = row["label"]
            image = None

            if pd.notna(image_name):
                matches = image_objects_by_name.get(image_name, [])
                if len(matches) == 1:
                    image = matches[0]
                else:
                    reason = (
                        f"Image name '{image_name}' is not unique."
                        if len(matches) > 1
                        else f"No image object matches image name '{image_name}'."
                    )
                    self._record_processing_error(
                        "curation_mapping",
                        ValueError(reason),
                        image_name=image_name,
                        cell_id=cell_id,
                    )
                    continue

            if image is None:
                frame = row.get("frame")
                matches = image_objects_by_frame.get(frame, [])
                if len(matches) == 1:
                    image = matches[0]
                else:
                    self._record_processing_error(
                        "curation_mapping",
                        ValueError(
                            f"Could not uniquely map curated cell {cell_id} to image "
                            f"(image_name={image_name!r}, frame={frame!r})."
                        ),
                        image_name=image_name if pd.notna(image_name) else None,
                        cell_id=cell_id,
                    )
                    continue

            frame = row.get("frame")
            if pd.notna(frame) and image.frame != frame:
                self._record_processing_error(
                    "curation_mapping",
                    ValueError(
                        f"Curated cell {cell_id} names image '{image.image_name}' "
                        f"but has frame {frame}; expected {image.frame}."
                    ),
                    image_name=image.image_name,
                    cell_id=cell_id,
                )
                continue

            cells = [cell for cell in image.cells if cell.cell_id == cell_id]
            if len(cells) != 1:
                reason = (
                    f"Cell ID {cell_id} was not found in image '{image.image_name}'."
                    if not cells
                    else f"Cell ID {cell_id} is not unique in image '{image.image_name}'."
                )
                self._record_processing_error(
                    "curation_mapping",
                    ValueError(reason),
                    image_name=image.image_name,
                    cell_id=cell_id,
                )
                continue
            if label == 0:
                image.cells.remove(cells[0])

        self._record_stage_success("curate_dataset")

    def curate_dataset(
        self, path_to_model, cols=4, control=False, save_curated_data=True
    ):
        """
        Curate the dataset based on a trained support vector machine model.

        This method calculates SVM features and uses a trained SVM model to classify cells based on these features.
        It curates the dataset by discarding cells that are classified as label 0 (non-interesting).
        Optionally, it can include positive and negative control cells.

        Parameters:
            path_to_model (str): Path to the trained SVM model.

            cols (int): Retained for backward compatibility; currently unused.

            control (bool): Whether to include positive and negative control cells.

            save_curated_data (bool): Whether to write the curated mesh pickle.
                The in-memory mesh dataframe is updated either way.

        Returns:
            None
        """
        self.batch_calculate_features(
            channel_method_pairs=[([None], "svm")], all_data=True, reset=True, max_mesh_size=800
        )

        from .curation import Curation

        cur = Curation(self.feature_dataframes["svm_None_features"])

        self.curated_df = cur.compiled_curation(path_to_model, cols)
  
        p1, p0 = cur.get_label_proportions()
        bactoscoop_logger.info(
            f"\nProportion of label 1: {p1}\nProportion of label 0: {p0}"
        )

        if control:
            import bactoscoop.plot as plotting

            self.pos, self.neg = cur.get_control(5, 5)
            if self.pos and self.neg is not None:
                bactoscoop_logger.info(
                    f"\nCell IDs and Frames for positive control are {self.pos}\nFrame and Cell IDs for negative control are {self.neg}"
                )
                plotting.plot_svm_controls(
                    self.pos, self.image_objects, message="Positive"
                )
                plotting.plot_svm_controls(
                    self.neg, self.image_objects, message="Negative"
                )

        self._apply_curation_labels()


        # The in-memory mesh population follows the curated cells even when
        # the caller is only testing and does not want a file overwritten.
        self.add_meshdata_to_dataframe()
        if save_curated_data:
            self._to_pickle(
                "{}_curated_meshdata.pkl".format(self.name), self.mesh_df_collection
            )

        del cur

    def add_meshdata_to_dataframe(self):
        data = []

        # Iterate over each image object
        for image in self.image_objects:
            # Iterate over each cell in the current image
            for cell in image.cells:
                # Extract the required attributes from the cell and image
                cell_id = cell.cell_id
                contour = cell.contour
                midline = cell.midline
                mesh = cell.mesh
                frame = image.frame
                image_name = image.image_name

                # Append the data as a dictionary to the list
                data.append(
                    {
                        "image_name": image_name,
                        "frame": frame,
                        "cell_id": cell_id,
                        "contour": contour,
                        "midline": midline,
                        "mesh": mesh,
                    }
                )

        # Create DataFrame from the collected data
        self.mesh_df_collection = pd.DataFrame(data)

    def export_curated_masks(self, output_subfolder="curated_masks", overwrite=True):
        """Write one label mask per image from the SVM-curated cell contours.

        This method is intended to be run after :meth:`curate_dataset`. It never
        changes the original Omnipose masks in ``masks``. Instead, it writes new
        masks to ``output_subfolder`` and records how every curated label overlaps
        the original labels, so the output can be used directly by BactoFeatures.

        Parameters
        ----------
        output_subfolder : str, optional
            Folder below ``image_folder_path`` for the exported label TIFFs.
        overwrite : bool, optional
            If ``False``, fail before replacing an existing curated-mask TIFF.

        Returns
        -------
        tuple[list[str], pandas.DataFrame]
            Paths to the exported masks and a provenance table with one row per
            curated cell.
        """
        self._require_image_folder_path()
        if not self.image_objects:
            raise ValueError(
                "No image objects are available. Load curated mesh data or run "
                "curate_dataset() before exporting curated masks."
            )

        output_dir = os.path.join(self.image_folder_path, output_subfolder)
        os.makedirs(output_dir, exist_ok=True)
        mask_paths = []
        provenance_rows = []

        for image in self.image_objects:
            if image.mask is None:
                raise ValueError(f"Original mask is not loaded for {image.image_name}.")

            max_label = len(image.cells)
            output_dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
            curated_mask = np.zeros(image.mask.shape, dtype=output_dtype)
            stem = os.path.splitext(image.image_name)[0]
            output_path = os.path.join(output_dir, f"{stem}_curated_masks.tif")
            if os.path.exists(output_path) and not overwrite:
                raise FileExistsError(
                    f"Curated mask already exists: {output_path}. Set overwrite=True to replace it."
                )

            for curated_label, cell in enumerate(image.cells, start=1):
                contour = np.asarray(cell.contour, dtype=float)
                if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
                    bactoscoop_logger.warning(
                        "Skipping curated cell %s in %s: invalid contour.",
                        cell.cell_id,
                        image.image_name,
                    )
                    continue

                cell_mask = np.zeros(image.mask.shape, dtype=np.uint8)
                # Cell contours are (row, column); OpenCV expects (x, y).
                polygon = np.rint(contour[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
                cv.fillPoly(cell_mask, [polygon], color=1)
                cell_pixels = cell_mask.astype(bool)
                if not cell_pixels.any():
                    bactoscoop_logger.warning(
                        "Skipping curated cell %s in %s: contour has no in-image pixels.",
                        cell.cell_id,
                        image.image_name,
                    )
                    continue

                curated_mask[cell_pixels] = curated_label
                source_labels = np.unique(image.mask[cell_pixels])
                source_labels = source_labels[source_labels != 0].astype(int).tolist()
                rows, columns = np.nonzero(cell_pixels)
                provenance_rows.append(
                    {
                        "image_name": image.image_name,
                        "frame": image.frame,
                        "curated_label": curated_label,
                        "bactoscoop_cell_id": cell.cell_id,
                        "centroid_x_px": float(columns.mean()),
                        "centroid_y_px": float(rows.mean()),
                        "area_px": int(cell_pixels.sum()),
                        "original_mask_path": os.path.join(
                            self.image_folder_path, "masks", f"{stem}_cp_masks.tif"
                        ),
                        "original_mask_labels": ";".join(map(str, source_labels)),
                        "curated_mask_path": output_path,
                    }
                )

            tifffile.imwrite(output_path, curated_mask)
            mask_paths.append(output_path)

        provenance = pd.DataFrame(provenance_rows)
        provenance_path = os.path.join(output_dir, "curated_mask_provenance.csv")
        provenance.to_csv(provenance_path, index=False)
        bactoscoop_logger.info(
            "Exported %s curated masks and %s curated cells to %s",
            len(mask_paths),
            len(provenance),
            output_dir,
        )
        return mask_paths, provenance

    def merge_dataframes(self, include_metadata_tag=False, discard_morphological_nan=False):
        """
        Merge feature dataframes into a single dataframe.
    
        Merges multiple feature dataframes into a single dataframe, ensuring that columns are properly renamed
        to identify the channel and removes duplicates based on specific columns.
    
        Args:
            include_metadata_tag (bool): Whether to rename 'image_name', 'cell_id', and 'frame' to 'Metadata_*'.
            discard_nan (bool): Whether to discard rows with NaN in 'cell_area' and 'cell_length'.
    
        Returns:
            pd.DataFrame: The merged feature dataframe.
        """
        if not self.feature_dataframes:
            raise ValueError("No feature dataframes are available to merge.")

        merged_dataframes = []
    
        for key, df in self.feature_dataframes.items():
            # Keys are "method_channel_features"; channel names may contain underscores.
            if not key.endswith("_features"):
                raise ValueError(f"Invalid feature dataframe key: {key!r}")
            method_and_channel = key[: -len("_features")]
            method, separator, channel = method_and_channel.partition("_")
            if not separator or not method or not channel:
                raise ValueError(f"Invalid feature dataframe key: {key!r}")
    
            # Set a prefix to identify the channel in column names if the channel is not None
            if channel != "None":
                df = df.rename(
                    columns={
                        col: f"{channel}_{col}"
                        for col in df.columns
                        if col not in ["image_name", "cell_id", "frame"]
                    }
                )
    
            # Append the modified dataframe to the list
            merged_dataframes.append(df)
    
        # Concatenate the dataframes vertically
        merged_data = pd.concat(merged_dataframes)
    
        # Merge the dataframes based on 'image_name', 'cell_id', and 'frame'
        result = merged_data.pivot_table(
            index=["image_name", "cell_id", "frame"], aggfunc="first"
        ).reset_index()
    
        # Discard rows with NaN in 'cell_area' and 'cell_length' if discard_nan is True
        if discard_morphological_nan:
            result = result.dropna(subset=["cell_area", "cell_length"])
    
        if include_metadata_tag:
            # Rename 'image_name', 'cell_id', and 'frame' to 'Metadata_*'
            result = result.rename(
                columns={
                    "image_name": "Metadata_image_name",
                    "cell_id": "Metadata_cell_id",
                    "frame": "Metadata_frame"
                }
            )

        # Reset the index to make it look like the final result
        self.merged_features = result.reset_index(drop=True)

        return self.merged_features

    def dataframe_to_pkl(self, pkl_name=None):
        """Save merged features in the image folder and return the pickle path.

        A suffixless ``pkl_name`` is a legacy tag appended to the dataset name;
        a name ending in ``.pkl`` is used as the exact filename.
        """
        if self.merged_features is None:
            raise ValueError("Merge feature dataframes before saving them.")
        if pkl_name is None:
            filename = f"{self.name}_features.pkl"
        elif str(pkl_name).endswith(".pkl"):
            filename = str(pkl_name)
        else:
            filename = f"{self.name}_{pkl_name}.pkl"
        self._to_pickle(filename, self.merged_features)
        return os.path.join(self.image_folder_path, filename)

    def meshdata_to_parquet(self, parquet_name=None):
        """Save the current mesh dataframe as reloadable Parquet in the image folder.

        Geometry arrays become nested numeric lists in Parquet. Reload with
        ``batch_load_mesh("name.parquet", phase_channel=...)``.
        Requires the optional ``pyarrow`` package.
        """
        if self.mesh_df_collection is None or self.mesh_df_collection.empty:
            raise ValueError("No mesh data is available to save.")
        name = parquet_name or f"{self.name}_meshdata.parquet"
        self._to_parquet(name, self.mesh_df_collection, mesh_data=True)

    def dataframe_to_parquet(self, parquet_name=None):
        """Save merged features as Parquet in the image folder (requires pyarrow)."""
        if self.merged_features is None:
            raise ValueError("Merge feature dataframes before saving them.")
        name = parquet_name or f"{self.name}_features.parquet"
        self._to_parquet(name, self.merged_features)

    def _atomic_write(self, filename, writer):
        self._require_image_folder_path()
        if os.path.basename(filename) != filename or filename in {"", ".", ".."}:
            raise ValueError("Output filename must be a name within the image folder.")
        file_path = os.path.join(self.image_folder_path, filename)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=self.image_folder_path
        )
        os.close(descriptor)
        try:
            writer(temporary_path)
            with open(temporary_path, "rb+") as temporary_file:
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, file_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _to_parquet(self, filename, data, mesh_data=False):
        if not filename.lower().endswith(".parquet"):
            raise ValueError("Parquet filename must end in .parquet")
        output = data.copy()
        if mesh_data:
            for column in ("contour", "mesh", "midline"):
                if column not in output:
                    raise ValueError(f"Mesh dataframe is missing '{column}'.")
                output[column] = output[column].map(
                    lambda value: np.asarray(value).tolist()
                )
        self._atomic_write(
            filename,
            lambda path: output.to_parquet(
                path, engine="pyarrow", compression="zstd", index=False
            ),
        )

    def _to_pickle(self, pkl_name, data):
        """
        Save data to a pkl file.

        """
        self._atomic_write(
            pkl_name,
            lambda path: self._write_pickle_file(path, data),
        )

    @staticmethod
    def _write_pickle_file(path, data):
        with open(path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
