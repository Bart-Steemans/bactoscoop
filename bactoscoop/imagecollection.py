# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:47:19 2023

@author: Bart Steemans. Govers Lab.
"""

import os
import pickle
from . import utilities as u

from tqdm import tqdm
from .image import Image
from .omni import Omnipose
from .curation import Curation
from .signalcorrelation import SignalCorrelation
import pandas as pd
import torch
import multiprocessing
from itertools import combinations

import bactoscoop.plot as plotting
import logging, sys
import gc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

bactoscoop_logger = logging.getLogger("logger")
# Change following to DEBUG to see more information
bactoscoop_logger.setLevel(logging.INFO)


class ImageCollection:
    """ """

    def __init__(self, image_folder_path=None, px=0.065):

        self.px = px  # microns per pixel
        self.images = None
        self.masks = None
        self.channel_images = {}
        self.inverted_image = None
        self.channel_list = None
        self.image_filenames = None

        self.image_folder_path = image_folder_path  # Path to the folder to be processed
        self.name = self.name = os.path.basename(
            os.path.normpath(self.image_folder_path)
        )

        self.image_objects = []  # Array that will be populated with objects

        self.mesh_df_collection = pd.DataFrame()

        self.feature_dataframes = {}

    def load_masks(self):
        """
        This method reads masks from the '/masks' folder within the image folder path and stores them in the 'self.masks' array.

        """
        self.masks, self.mask_filenames = u.read_tiff_folder(
            self.image_folder_path + "/masks"
        )

    def load_phase_images(self, phase_channel=""):
        """
        This method loads phase contrast images from the specified image folder path, based on the characters before the file-extension,
        and stores them in the 'self.images' array.

        Parameters
        ----------
        phase_channel : str, optional
            Characters before the file-extension (e.g., "c1", "phase") to look for when using this function. (default is an empty string)

        """
        self.images, self.image_filenames, self.paths = u.read_tiff_folder(
            self.image_folder_path, phase_channel, include_paths=True
        )

    def load_channel_images(self, channel_list):
        """
        This method loads channel images from the image folder path based on the specified characters before the file-extension and stores them in
        the 'self.channel_images' dictionary, where the channel name is used as the key.

        Parameters
        ----------
        channel_list : List of str
            Characters before the extension (e.g., ["c2", "c3"]) to look for when using this function.

        """
        self.channel_list = channel_list

        self.channel_images = u.read_channels(
            self.image_folder_path, self.channel_images, self.channel_list
        )

    def create_image_objects(self, phase_channel=""):
        """
        This method creates image objects from loaded phase images and masks and stores them in the 'image_objects' list.
        If masks and phase images are not already loaded, this method will load them automatically.

        Parameters
        ----------
        phase_channel : str, optional
            Characters before the file-extension to look for when loading phase images. (default is an empty string)
        """
        if self.masks is None:
            self.load_masks()
            self.load_phase_images(phase_channel)
            
        assert len(self.masks) == len(self.images), "Lengths of masks and images should be the same."

        self.image_objects = []
        
        for i, image_name in enumerate(self.image_filenames):

            image = self.images[i]
            mask = self.masks[i]
            mesh_df = self.get_mesh_dataframe(image_name)

            img_obj = self.create_image_object(
                image, image_name, i, mask, mesh_df, self.px
            )

            # self.add_channels(img_obj, self.channel_images, i) # Loads all channels at once, maybe we only want to load them when needed

            self.image_objects.append(img_obj)

    def add_channels(self, img_objects, channel_list):
        """
        This method loads channel images based on the provided channel list and adds them to the individual image objects.

        Parameters
        ----------
        img_objects : List
            List of image objects.
        channel_list : List
            List of channel names.

        """
        if channel_list is None:
            raise ValueError(
                "No channel_list provided. It should be of format ['channelname1','channelname2']."
            )

        # Load channel images based on the provided channel list
        self.load_channel_images(channel_list)

        for i, img_obj in enumerate(img_objects):
            channel_single_image_dict = {}
            if (
                self.channel_images is not None
            ):  # Check if self.channel_images is not None
                for channel, images in self.channel_images.items():
                    channel_single_image_dict[channel] = images[i]

                img_obj.channels = channel_single_image_dict

    def get_mesh_dataframe(self, image_name):
        """
        This method returns the mesh dataframe associated with the specified image name.

        Parameters
        ----------
        image_name : str
            The name or identifier of the image.

        """

        if not self.mesh_df_collection.empty:
            return self.mesh_df_collection[
                self.mesh_df_collection["image_name"] == image_name
            ].reset_index(drop=True)
        else:
            return None

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

        img_obj = Image(image, image_name, index, mask, mesh_df)

        if mesh_df is not None:
            img_obj.create_cell_object(verbose=False)
        return img_obj

    # Methods for batch processing of images ----------------------------------------------
    def segment_images(self, mask_thresh, minsize, n, model_name="bact_phase_omni"):
        """
        Segment phase contrast images in the folder using a pretrained omnipose model.

        Parameters
        ----------
        mask_thresh : float
            A parameter to control mask thickness. Increasing makes the mask thinner, decreasing makes it larger.
        minsize : int
            Minimum size of masks to be kept.
        n : int
            The number of images to segment.
        model_name : str, optional
            The name of the pretrained omnipose model to be used for segmentation. (default is 'bact_phase_omni')

        """
        omni = Omnipose(self.images, self.paths)
        omni.load_models(model_name)

        omni.compiled_process(n, mask_thresh, minsize)
        torch.cuda.empty_cache()

        del omni
        gc.collect()

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
        if reset_channels:
            self.channel_images = {}

        channels_to_load = [
            channel for channel in channels if channel not in self.channel_images.keys()
        ]
        if channels_to_load:
            self.add_channels(self.image_objects, channels_to_load)

        bactoscoop_logger.info("Detecting objects within cell ...")

        dfs = (
            image.object_detection(
                channels,
                smoothing,
                align,
                log_sigma,
                kernel_width,
                min_overlap_ratio,
                max_external_ratio,
            )
            for image in tqdm(self.image_objects)
        )

        self.object_detection_df = pd.concat(dfs, axis=0)

        bactoscoop_logger.info(self.object_detection_df["channel"].value_counts())

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

        """
        self.mesh_df_collection = pd.DataFrame()

        if isinstance(object_list, Image):
            print("\n")

            bactoscoop_logger.info(f"PROCESSING IMAGE: {object_list.image_name}")

            object_list.join_split_pipeline(
                join_thresh, split_thresh, CD_width, smoothing
            )

            self.mesh_df_collection = getattr(object_list, "processed_mesh_dataframe")

            object_list.create_cell_object(verbose=False)

        else:

            if object_list is None:
                self.create_image_objects(phase_channel=phase_channel)

                object_list = self.image_objects

            for img_obj in object_list:
                print("\n")
                bactoscoop_logger.info(f"PROCESSING IMAGE: {img_obj.image_name}")

                img_obj.join_split_pipeline(
                    join_thresh, split_thresh, CD_width, smoothing
                )

                self.mesh_df_collection = pd.concat(
                    [
                        self.mesh_df_collection,
                        getattr(img_obj, "processed_mesh_dataframe"),
                    ]
                )

                img_obj.create_cell_object(verbose=False)

        if save_data:

            if pkl_name is not None:

                self._to_pickle(pkl_name, self.mesh_df_collection)

            self._to_pickle(
                "{}_meshdata.pkl".format(self.name), self.mesh_df_collection
            )

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

        with open(os.path.join(pkl_path, pkl_name), "rb") as f:
            self.mesh_df_collection = pickle.load(f)

        bactoscoop_logger.info(f"Meshes are loaded from a pickle file named {pkl_name}")

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

        if isinstance(object_list, Image):
            print("\n")
            bactoscoop_logger.info(f"PROCESSING IMAGE: {object_list.image_name}")

            object_list.mask2mesh(smoothing)
            self.mesh_df_collection = getattr(object_list, "mesh_dataframe")
            object_list.create_cell_object(verbose=False)
        else:

            if object_list is None:
                self.create_image_objects(phase_channel=phase_channel)
                object_list = self.image_objects

            for img_obj in object_list:

                bactoscoop_logger.info(f"PROCESSING IMAGE: {img_obj.image_name}")

                img_obj.mask2mesh()
                self.mesh_df_collection = pd.concat(
                    [self.mesh_df_collection, getattr(img_obj, "mesh_dataframe")]
                )

                img_obj.create_cell_object(verbose=False)

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

        if reset_channels:
            self.channel_images = {}

        channels_to_load = [
            channel
            for channel in shifted_channel
            if channel not in self.channel_images.keys()
        ]
        if channels_to_load:
            self.add_channels(self.image_objects, channels_to_load)

        success_percentage_list = []
        bactoscoop_logger.info("Aligning contour, mesh and midline to channel image ...")
        for image_obj in tqdm(self.image_objects):
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
            success_percentage_list.append(success_percentage)
        average_success_percentage = sum(success_percentage_list) / len(
            success_percentage_list
        )

        bactoscoop_logger.info(f"Shift succes rate: {average_success_percentage}")

    def batch_calculate_features(
        self,
        channel_method_pairs,
        all_data=True,
        reset=True,
        use_shifted_contours=False,
        shift_signal=False,
        max_mesh_size=600,
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

        Returns
        -------
        Dict
            A dictionary containing dataframes with calculated features.

        """
        if reset:
            self.feature_dataframes = {}

        for channels, method in channel_method_pairs:
            for channel in channels:
                bactoscoop_logger.info(f"PROCESSING CHANNEL: {channel}, method: {method}")
                feature_dfs = []

                for img_obj in tqdm(self.image_objects):
                    # Calculate features for the specified channel and method
                    result_df = img_obj.calculate_features(
                        method,
                        channel,
                        all_data,
                        use_shifted_contours,
                        shift_signal,
                        max_mesh_size,
                    )
                    feature_dfs.append(result_df)

                concatenated_df = pd.concat(feature_dfs, axis=0)

                concatenated_df.reset_index(drop=True, inplace=True)

                self.feature_dataframes[f"{method}_{channel}_features"] = (
                    concatenated_df
                )

        return self.feature_dataframes

    def batch_calculate_signal_correlation_features(
        self, df, channels, feature_method_tuples
    ):
        # Generate all unique pairs of channels
        channel_pairs = combinations(channels, 2)

        for channel1, channel2 in channel_pairs:
            for features, method_names in feature_method_tuples:
                bactoscoop_logger.info(
                    f"Processing channel {channel1} vs {channel2} and features {features}:"
                )
                for feature in tqdm(features):

                    for method_name in method_names:
                        sc = SignalCorrelation(
                            df, channel1, channel2, feature, method_name
                        )
                        sc.calculate()
                        del sc
        return df

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

            cols (int): Number of columns to exclude during curation.

            control (bool): Whether to include positive and negative control cells.

        Returns:
            None
        """
        self.batch_calculate_features(
            channel_method_pairs=[([None], "svm")], all_data=True, reset=True
        )

        cur = Curation(self.feature_dataframes["svm_None_features"])

        self.curated_df = cur.compiled_curation(path_to_model, cols)

        p1, p0 = cur.get_label_proportions()
        bactoscoop_logger.info(
            f"\nProportion of label 1: {p1}\nProportion of label 0: {p0}"
        )

        if control:
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

        for index, row in self.curated_df.iterrows():
            frame = row["frame"]
            cell_id = row["cell_id"]
            label = row["label"]

            # Find the Image object in the list based on the 'frame' index
            image = self.image_objects[frame]

            # Find the Cell object in the Image object based on the 'cell_id' index
            cell = next((c for c in image.cells if c.cell_id == cell_id), None)

            if cell is not None and label == 0:
                # Remove the Cell object from the list of Cells within the Image object
                image.cells.remove(cell)

        if save_curated_data:
            self.add_meshdata_to_dataframe()
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
        merged_dataframes = []
    
        for key, df in self.feature_dataframes.items():
            # Split the key into its parts
            parts = key.split("_")
            channel = parts[1]  # Extract the channel
    
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
        """
        Save the merged_features dataframe to a pickle-file.

        """
        if self.merged_features is not None:

            if pkl_name is not None:

                self._to_pickle(
                    "{}_{}.pkl".format(self.name, pkl_name), self.merged_features
                )
            else:
                self._to_pickle(
                    "{}_features.pkl".format(self.name), self.merged_features
                )

    def _to_pickle(self, pkl_name, data):
        """
        Save data to a pkl file.

        """
        file_path = os.path.join(self.image_folder_path, pkl_name)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)


class Pipeline:

    def __init__(self):
        self.exp_folder_path = None

    def process_image_folder(self, image_path, kwargs):
        channel1 = kwargs.get("channel1", "")
        channel2 = kwargs.get("channel2", "")
        channel3 = kwargs.get("channel3", "")
        ic = ImageCollection(image_path)

        if "segment" in kwargs and kwargs["segment"]:
            ic.load_phase_images(phase_channel=channel1)
            i = len(ic.images)
            ic.segment_images(
                kwargs.get("mask_thresh", 1), kwargs.get("minsize", 300), n=range(i)
            )

        if "load_mesh" in kwargs and kwargs["load_mesh"]:
            filename = ic.name + "_meshdata.pkl"
            ic.batch_load_mesh(filename, phase_channel=channel1)
        elif "load_curated_mesh" in kwargs and kwargs["load_curated_mesh"]:
            filename = ic.name + "_curated_meshdata.pkl"
            ic.batch_load_mesh(filename, phase_channel=channel1)
        else:
            ic.batch_process_mesh(
                object_list=None,
                phase_channel=channel1,
                smoothing=kwargs.get("smoothing_mesh", 0.1),
                join_thresh=kwargs.get("join_thresh", 4),
                split_thresh=kwargs.get("split_thresh", 0.4),
                CD_width=kwargs.get("CD_width", False),
            )
            del ic.images
            del ic.masks
        if "save_svm_dataframe" in kwargs and kwargs["save_svm_dataframe"]:
            ic.batch_calculate_features(
                channel_method_pairs=[([None], "svm")],
                all_data=False,
                reset=True,
                max_mesh_size=1000,
            )
            ic.merge_dataframes()
            ic.dataframe_to_pkl("svm_features")
        if "curation" in kwargs and kwargs["curation"] is not None:
            ic.curate_dataset(kwargs["curation"], save_curated_data=True)

        if "detect_objects" in kwargs and kwargs["detect_objects"] is not None:
            ic.batch_detect_objects(
                channels=kwargs.get("channels_detect_objects", ""),
                smoothing=kwargs.get("smoothing_objects", 0.1),
                reset_channels=kwargs.get("reset_channels_object_detection", True),
                align = kwargs.get("align_for_object_detection", False),
                log_sigma=kwargs.get("log_sigma", 3),
                kernel_width=kwargs.get("kernel_width", 4),
                min_overlap_ratio=kwargs.get("min_overlap_ratio_detect_objects", 0.001),
                max_external_ratio=kwargs.get("max_external_ratio_detect_objects", 0.2),
            )
        if "detect_objects_with_membrane" in kwargs and kwargs["detect_objects_with_membrane"] is not None:
            ic.batch_detect_objects(
                channels=kwargs.get("channels_detect_objects", ['C2']),
                smoothing=kwargs.get("smoothing_objects", 0.1),
                reset_channels=True,
                align = True,
                log_sigma=kwargs.get("log_sigma", 3),
                kernel_width=kwargs.get("kernel_width", 4),
                min_overlap_ratio=kwargs.get("min_overlap_ratio_detect_objects", 0.001),
                max_external_ratio=kwargs.get("max_external_ratio_detect_objects", 0.4),
            )
            ic.batch_detect_objects(
                channels=kwargs.get("channels_detect_objects", ['C3', 'C4', 'C5']),
                smoothing=kwargs.get("smoothing_objects", 0.1),
                reset_channels=False,
                align = False,
                log_sigma=kwargs.get("log_sigma", 3),
                kernel_width=kwargs.get("kernel_width", 4),
                min_overlap_ratio=kwargs.get("min_overlap_ratio_detect_objects", 0.001),
                max_external_ratio=kwargs.get("max_external_ratio_detect_objects", 0.4),
            )

        if "shift_contours" in kwargs and kwargs["shift_contours"] is not None:
            ic.batch_shift_correction(
                shifted_channel=kwargs.get("shifted_channel", [channel2]),
                reset_channels=True,
                log_sigma=kwargs.get("log_sigma_shift", 1.5),
                kernel_width=kwargs.get("kernel_width_shift", 3),
                min_overlap_ratio=kwargs.get("min_overlap_ratio_detect_shift", 0.001),
                max_external_ratio=kwargs.get("max_external_ratio_detect_shift", 0.9),
                phase_log_sigma=kwargs.get("phase_log_sigma_shift", 0.5),
                phase_closing_level=2,
                signal_closing_level=12,
            )

        if "calculate_features" in kwargs and kwargs["calculate_features"] is not None:
            feature_type = kwargs.get("chann_method_tuple", "")
            ic.batch_calculate_features(
                channel_method_pairs=feature_type,
                all_data=kwargs.get("all_data", False),
                reset=True,
                use_shifted_contours=kwargs.get("use_shifted_contours", False),
                max_mesh_size=1000,
            )
            ic.merge_dataframes(include_metadata_tag=True, discard_morphological_nan=True)
            ic.dataframe_to_pkl(pkl_name=kwargs.get("pkl_ext", None))
            
        if "calculate_features_with_membrane" in kwargs and kwargs["calculate_features_with_membrane"] is not None:
            feature_type = kwargs.get("chann_method_tuple", "")
            ic.batch_calculate_features(
                channel_method_pairs=[([None], "morphological"), (['C2'], "objects"), (['C2'], "membrane"), (['C2'], "profiling")],
                all_data=kwargs.get("all_data", False),
                shift_signal=True,
                reset=True,
                use_shifted_contours=kwargs.get("use_shifted_contours", False),
                max_mesh_size=800,
            )
            feature_type = kwargs.get("chann_method_tuple", "")
            ic.batch_calculate_features(
                channel_method_pairs=[(['C3', 'C4', 'C5'], "objects"), (['C3'], "membrane"), (['C3', 'C4', 'C5'], "profiling")],
                all_data=kwargs.get("all_data", False),
                shift_signal=False,
                reset=False,
                use_shifted_contours=kwargs.get("use_shifted_contours", False),
                max_mesh_size=800,
            )
            ic.merge_dataframes(include_metadata_tag=True, discard_morphological_nan=True)
        if "calculate_correlation" in kwargs and kwargs["calculate_correlation"] is not None:
            feature_method_tuples = [
                (['normalized_axial_intensity', 'normalized_average_mesh_intensity', 'radial_intensity_distribution'], ['manders', 'pearson', 'li_icq', 'spearman','kendall', 'distance_corr','covariance', 'n_cross_corr','entropy_diff', 'kurtosis_ratio', 'skewness_product', 'zero_crossings_diff', 'fft_peak_ratio', 'fft_energy_ratio', 'histogram_intersection', 'cosine_similarity']),
                (['cell_total_obj_area'], ['ratio']),
            ]

            # Calculate signal correlation features
            ic.batch_calculate_signal_correlation_features(ic.merged_features, ['C2', 'C3', 'C4', 'C5'], feature_method_tuples = feature_method_tuples)
            feature_method_tuples = [
                (["normalized_contour_intensity", "complemented_contour_intensity"], ['manders', 'pearson', 'li_icq', 'spearman','kendall', 'distance_corr','covariance', 'n_cross_corr','entropy_diff', 'kurtosis_ratio', 'skewness_product', 'zero_crossings_diff', 'fft_peak_ratio', 'fft_energy_ratio', 'histogram_intersection', 'cosine_similarity'])
            ]
            ic.batch_calculate_signal_correlation_features(ic.merged_features, ['C2', 'C3'], feature_method_tuples = feature_method_tuples)

        ic.dataframe_to_pkl(pkl_name=kwargs.get("pkl_ext", None))
            

        del ic
        gc.collect()

    def general_pipeline_parallel(self, exp_folder_path, **kwargs):
        self.exp_folder_path = exp_folder_path
    
        subfolder_paths = [
            os.path.join(exp_folder_path, directory)
            for directory in os.listdir(exp_folder_path)
            if os.path.isdir(os.path.join(exp_folder_path, directory))
        ]
        print(f"\nProcessed folders are: {subfolder_paths}")
    
        num_cores = kwargs.get("n_cores", 2)
        
        # Use try-finally to ensure processes are cleaned up
        pool = multiprocessing.Pool(num_cores)
        
        try:
            pool.starmap(
                self.process_image_folder,
                [(image_path, kwargs) for image_path in subfolder_paths],
            )
        finally:
            pool.close()  # Prevents new tasks from being submitted
            pool.join()   # Waits for all worker processes to finish
            pool.terminate()  # Kills any remaining processes
            del pool  # Explicitly delete pool object
            gc.collect()  # Force garbage collection

    def general_pipeline_sequential(self, exp_folder_path, svm=False, **kwargs):
        self.exp_folder_path = exp_folder_path

        subfolder_paths = [
            os.path.join(exp_folder_path, directory)
            for directory in os.listdir(exp_folder_path)
            if os.path.isdir(os.path.join(exp_folder_path, directory))
        ]
        print(f"\nProcessed folders are: {subfolder_paths}")
        if svm:
            print("SVM_features")
            for image_path in subfolder_paths:
                self.process_image_folder_for_svm(image_path, kwargs)
        else:
            for image_path in subfolder_paths:
                self.process_image_folder(image_path, kwargs)
