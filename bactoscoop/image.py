# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:45:53 2023

@author: Bart Steemans. Govers Lab.
"""
from .features import Features
from . import utilities as u

from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import cv2 as cv
import logging

warnings.filterwarnings("ignore")
bactoscoop_logger = logging.getLogger("logger")


class Image:

    def __init__(
        self,
        image,
        image_name,
        frame,
        mask=None,
        processed_mesh_dataframe=None,
        px=0.065,
    ):
        """
        Initialize an instance of your class.

        Parameters
        ----------
        image : phase contrast image
            The phase contrast image associated with this instance.
        image_name : str
            The name or identifier of the image.
        frame : int
            The frame number
        mask : mask, optional
            The segmented image.
        processed_mesh_dataframe : pd.DataFrame, optional
            A processed mesh dataframe if available. (default is None)
        px : float, optional
        The pixel size in micrometers. (default is 0.065)
        """

        self.mask = mask
        self.cells = []  # List of cell objects
        self.px = px  # micron per pixel
        self.image_name = image_name
        self.frame = frame
        self.image = image
        self.channels = None  # Dictionary of the color channels

        self.joined_mask = None  # Mask images after cell joining
        self.mesh_dataframe = (
            None  # Resulting dataframe containing the meshes of the cells
        )
        self.processed_mesh_dataframe = processed_mesh_dataframe

        self.im_interp2d = None  # 2D interpolation of the phase contrast image
        self.bg_channels = (
            {}
        )  # Dictionary of the 2D interpolations of the different color channels
        self.inverted_image = None  # Placeholder for the inverted phase contrast image

    def calculate_features(
        self,
        method,
        channel,
        all_data,
        use_shifted_contours,
        shift_signal,
        max_mesh_size,
    ):
        """
        Calculate features for cells based on the specified method and channel.

        This method calculates features for individual cells in the image based on the provided method
        and channel. It also allows for adding profiling data if required.

        Parameters
        ----------
        method : str
            The method used to calculate features for cells.
            The methods are: 'morphological', 'profiling', 'svm', 'objects', 'phase'. (To be expanded)
        channel : str
            The channel for which features are calculated.
        add_profiling_data : bool
            Whether to include profiling data in the feature calculations.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the calculated features for the cells.
        """
        features = Features(self)
        result_df = None
        bad_celllist = []
        
        # First, filter out cells that should be discarded by discard_large_meshes
        filtered_cells = []
        for cell in self.cells:
            cell.discard = False
            features.discard_large_meshes(cell, max_mesh_size)
            
            if not cell.discard:
                filtered_cells.append(cell)
        
        # Update self.cells to only contain non-discarded cells
        self.cells = filtered_cells
        
        # Now, process the remaining cells
        for cell in self.cells:
            feature_calculation_method = f"{method}"
                
            if hasattr(features, feature_calculation_method):
                try:
                    calculation_method = getattr(features, feature_calculation_method)
        
                    calculation_method(
                        cell, channel, all_data, use_shifted_contours, shift_signal
                    )
        
                except Exception as e:
                    bactoscoop_logger.debug(f"Encountered an Exception: {e}")
                    bad_celllist.append(cell.cell_id)
                    cell.discard = True
            if feature_calculation_method == "svm":
                
                filtered_cells = []
                for cell in self.cells:
                    if not cell.discard:
                        filtered_cells.append(cell)
                
                # Update self.cells to only contain non-discarded cells
                self.cells = filtered_cells
        if bad_celllist:
            bactoscoop_logger.debug(
                f"Unable to calculate {method} features from cells: {bad_celllist} in frame {self.frame}"
            )
        result_df = self.features_to_dataframe(method)
        return result_df

    # def calculate_signal_correlation_features(
    #     self,
    #     method,
    #     signal1,
    #     signal2,
    #     shift_signal1,
    #     shift_signal2,
    #     all_data,
    #     use_shifted_contours,
    #     max_mesh_size,
    # ):

    #     features = Features(self)
    #     result_df = None
    #     bad_celllist = []
    #     for cell in self.cells:
    #         cell.discard = False
    #         features.discard_large_meshes(cell, max_mesh_size)

    #         if not cell.discard:
    #             feature_calculation_method = f"{method}"
    #             if hasattr(features, feature_calculation_method):
    #                 try:
    #                     calculation_method = getattr(
    #                         features, feature_calculation_method
    #                     )

    #                     calculation_method(
    #                         cell,
    #                         signal1,
    #                         signal2,
    #                         shift_signal1,
    #                         shift_signal2,
    #                         all_data,
    #                         use_shifted_contours,
    #                     )

    #                 except Exception as e:
    #                     # print(f"Encountered a ValueError: {e}")
    #                     bad_celllist.append(cell.cell_id)
    #                     cell.discard = True

    #     if bad_celllist:
    #         bactoscoop_logger.debug(
    #             f"Unable to calculate {method} features from cells: {bad_celllist} in frame {self.frame}"
    #         )
    #     result_df = self.features_to_dataframe(method)
    #     return result_df

    def features_to_dataframe(self, method):
        """
        Convert cell features stored in a dictionary to a pandas DataFrame.

        This method takes a dictionary of cell features, specified by the method name, and converts it into a pandas DataFrame.
        Additionally, it adds information features to the DataFrame.

        Parameters
        ----------
        method : str
            The name of the method used for feature calculation.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing cell features.
        """
        features_df = self.dataframe_from_dict(f"{method}_features")

        self.add_info_feature_df(features_df)

        return features_df

    def dataframe_from_dict(self, feature_dictionary):
        """
        Convert a feature dictionary to a pandas DataFrame.

        This method takes a feature dictionary, which maps cell IDs to feature values, and converts it into a pandas DataFrame.

        Parameters
        ----------
        feature_dictionary : str
            The name of the feature dictionary to convert.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing cell features.
        """
        data = {
            cell.cell_id: getattr(cell, feature_dictionary)
            for cell in self.cells
            if not cell.discard
        }

        df = pd.DataFrame.from_dict(data, orient="index")

        return df

    def add_info_feature_df(self, dataframe):
        """
        Add information columns to a feature DataFrame.

        This method adds information columns, including 'image_name', 'cell_id', and 'frame',
        to an existing DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to which information columns will be added.
        """

        dataframe.insert(0, "image_name", self.image_name)
        dataframe.insert(
            0, "cell_id", [cell.cell_id for cell in self.cells if not cell.discard]
        )
        dataframe.insert(0, "frame", self.frame)

    def join_split_pipeline(
        self, join_thresh=4, split_thresh=0.3, CD_width=False, smoothing=0.1
    ):
        """
        Perform a pipeline for joining and splitting cells.

        This method is used within the batch_mesh_process() method of ImageCollection. It involves joining cells based on
        a specified threshold, converting masks to mesh data, splitting cells based on a specified threshold,
        and creating cell objects.

        Parameters
        ----------
        join_thresh : int, optional
            The threshold for joining cells in pixels. (default is 4)
        split_thresh : float, optional
            The threshold for splitting cells. (default is 0.3)

        Returns
        -------
        None
        """
        self.join_cells(join_thresh, smoothing)
        self.mask2mesh(smoothing)
        self.split_cells(split_thresh, CD_width)
        self.create_cell_object()

    def create_cell_object(self, verbose=True):
        """
        Create Cell objects from mesh data.

        This method creates Cell objects based on mesh data contained in a DataFrame. The Cell objects are
        initialized with contour, mesh, and midline data, and they are added to the list of cells in the Image object.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print a message indicating the source of the Cell objects (processed or unprocessed meshes). (default is True)

        Returns
        -------
        None
        """
        self.cells = []

        if self.processed_mesh_dataframe is None:
            try:
                dataframe = self.mesh_dataframe
                message_prefix = "Cell objects are created from unprocessed meshes"
            except AttributeError:
                bactoscoop_logger.info(
                    "No meshes from the masks. Either process the masks or load in mesh data from a .pkl file."
                )
                return
        else:
            dataframe = self.processed_mesh_dataframe
            message_prefix = "Cell objects are created from processed meshes"

        for i, row in dataframe.iterrows():
            if "cell_id" not in row or pd.isna(row["cell_id"]):
                cell_ID = i
            else:
                cell_ID = row["cell_id"]
            contours = np.array(row["contour"])
            meshes = np.array(row["mesh"])
            midlines = np.array(row["midline"])
            cell = Cell(contours, meshes, midlines, self.image.shape, cell_ID)
            self.add_cell(cell)

        num_cells_created = len(self.cells)
        message = f"{message_prefix}. {num_cells_created} Cell objects are created."

        if verbose:
            print(message)

    def add_cell(self, cell):
        """
        Add a Cell object to the list of cells in the Image object.
        """
        self.cells.append(cell)

    def get_inverted_image(self):
        """
        Calculate the inverted image for phase contrast.

        This method computes the inverted image for phase contrast imaging. It subtracts the maximum pixel value
        from the original image and does a background subtraction (bgr = mean(background intensity)) to obtain the inverted image.

        """
        inverted_image = np.subtract(np.max(self.image), self.image)
        bgr = u.median_background(inverted_image, self.mask)  #
        self.inverted_image = np.maximum(inverted_image - bgr, 0)

    def get_bg_channel(self, channel):
        bg_function, bg_matrix = u.construct_bg_function(
            self.channels[channel], self.mask
        )
        self.bg_channels[channel] = np.maximum(self.channels[channel] - bg_matrix, 0)

    def channel_interp2d(self):
        """
        Interpolate channel images using 2D spline interpolation.

        This method interpolates the channel images using 2D spline interpolation and stores the interpolated images
        in the 'chann_interp2d' dictionary.
        """
        for channel in self.channels:
            spline = u.interp2d(self.channels[channel])
            self.chann_interp2d[channel] = spline

    def join_cells(self, thresh=None, smoothing=0.1):
        """
        Join cells based on pole-to-pole distance.

        This method attempts to join adjacent cells within the image based on the distance between their poles. It identifies cell pairs
        with pole-to-pole distances below a specified threshold and merges them into a single cell. The joined cell IDs are updated in the
        'joined_mask' array.

        Parameters
        ----------
        thresh : float, optional
            The distance threshold for joining cells. Cells with pole-to-pole distances below this value will be merged. If None, no
            threshold is applied, and all cells will be considered for joining. (default is None)

        """

        if self.mask is None:
            raise ValueError("Mask image is not loaded")
        bactoscoop_logger.info(">>> Joining cells ...")

        pole_1 = []
        pole_2 = []
        cell_ID = []

        for cell in tqdm(range(1, np.max(self.mask))):
            temp_mask = np.where(self.mask == cell, 1, 0)
            temp_mask = cv.convertScaleAbs(temp_mask)

            _, cropped_mask, _, x, y = u.crop_image(mask_to_crop=temp_mask)

            cell_contours = u.get_object_contours(cropped_mask, smoothing=smoothing)
            for cn in cell_contours:
                try:
                    skeleton = u.extract_skeleton(cropped_mask)
                except Exception:
                    skeleton = []
                    cell_ID.append(cell)
                    continue
                if np.any(skeleton):
                    extended_skeleton, pole1, pole2 = u.extend_skeleton(
                        skeleton, cn, find_pole1=True, find_pole2=True
                    )
                    pole1 += np.array([y, x])
                    pole2 += np.array([y, x])
                    if pole1.shape == (1, 2) and pole2.shape == (1, 2):
                        pole_1.append(pole1)
                        pole_2.append(pole2)
                    else:
                        cell_ID.append(cell)
                else:
                    cell_ID.append(cell)

        if pole_1 == []:
            cell_pairs = []
        else:
            pole_1 = np.vstack(np.array(pole_1))
            pole_2 = np.vstack(np.array(pole_2))

            cell_pairs = u.get_cell_pairs(pole_1, pole_2, cell_ID, thresh)

        if not np.any(cell_pairs):
            self.joined_mask = np.copy(self.mask)
        else:
            # Create a copy of the original masks array to store the new masks
            new_masks = np.copy(self.mask)
            dsu = {}
            for mask in np.unique(cell_pairs):
                dsu[mask + 1] = mask + 1
            # Merge masks that belong to the same group
            for mask1, mask2 in cell_pairs:
                root1 = mask1 + 1
                while root1 != dsu[root1]:
                    root1 = dsu[root1]
                root2 = mask2 + 1
                while root2 != dsu[root2]:
                    root2 = dsu[root2]
                if root1 != root2:
                    dsu[root1] = root2

            # Update the masks with the merged values
            for mask, root in dsu.items():
                new_masks[new_masks == mask] = root
            unique_masks = np.unique(new_masks)
            # Create a dictionary that maps unique mask values to new mask values starting from 1
            mask_dict = {mask: i for i, mask in enumerate(unique_masks)}

            # Use the dictionary to relabel the new_masks array and the new_labels array
            new_masks_relabel = np.vectorize(mask_dict.get)(new_masks)
            self.joined_mask = new_masks_relabel

    def mask2mesh(self, smoothing=0.1):
        """
        Create meshes from cell masks.

        This method processes cell masks to create corresponding meshes. It generates contours and skeletons for each cell in the mask,
        extends the skeleton, and calculates additional geometric properties, such as length and width. Based on these properties, it
        straightens the cell contour and creates a mesh representation. The resulting mesh data, including contour, mesh, and midline,
        is stored in a DataFrame and associated with the image frame.

        This method is based on the cell profile creation of the MOMIA package (doi: 10.1016/j.celrep.2021.110154)

        """
        if self.joined_mask is not None:
            mask = self.joined_mask
            bactoscoop_logger.info(">>> Creating meshes from joined masks ...")
        else:
            mask = self.mask
            bactoscoop_logger.info(">>> Creating meshes from raw masks ...")
        # loop through all cells in the frame
        self.mesh_dataframe = u.get_cellular_mesh(mask, smoothing=smoothing)
        self.mesh_dataframe["frame"] = self.frame
        self.mesh_dataframe["image_name"] = self.image_name

    def split_cells(self, thresh=0.3, CD_width=False):
        """
        Split cells based on a constriction degree threshold.

        This method separates cells within the image based on a given constriction degree threshold. It calculates the constriction degree
        for each cell and, if it exceeds the threshold, splits the cell into two separate cells. The new mesh data, including contour, mesh,
        and midline, is stored in a DataFrame and associated with the image frame.

        This method is based on the splitting algorithm in the Oufti software package (doi: 10.1111/mmi.13264).

        Parameters
        ----------
        thresh : float, optional
            The constriction degree threshold for splitting cells. Cells with a constriction degree greater than this threshold will be split.
            The default is 0.3.

        """
        self.get_inverted_image()
        im_interp2d = u.interp2d(self.inverted_image)

        bactoscoop_logger.info(">>> Splitting cells ...")

        new_meshes = []
        new_contours = []
        new_midlines = []

        for j in tqdm(range(len(self.mesh_dataframe))):

            x1, y1, x2, y2, contour, midline = u.separate_singleframe_meshdata(
                j, self.mesh_dataframe
            )

            complete_mesh = np.array(np.column_stack((x1, y1, x2, y2)))

            step_length_px = u.get_step_length(x1, y1, x2, y2, self.px)

            width_px = u.get_width(x1, y1, x2, y2) * self.px
            if CD_width:
                weighted_intprofile = width_px
            else:
                intprofile = u.measure_smoothened_intensity(
                    midline, im_interp2d, width=5
                )
                weighted_intprofile = u.get_weighted_intprofile(intprofile, width_px)

            constrDegree, relPos, constrDegree_abs, ctpos = (
                u.constr_degree_single_cell_min(
                    weighted_intprofile,
                    step_length_px,
                    upper_limit=1.0,
                    lower_limit=0.0,
                )
            )

            if 0.95 > constrDegree > thresh:
                try:
                    x, y, left, right = u.split_point(x1, y1, x2, y2, ctpos)

                    mesh1rec = np.concatenate(
                        (complete_mesh[: (ctpos - 2)], right, np.array([[x, y, x, y]]))
                    )
                    mesh2rec = np.concatenate(
                        (np.array([[x, y, x, y]]), left, complete_mesh[(ctpos + 2) :])
                    )

                    mesh1, contour1, midline1 = u.split_mesh2mesh(
                        mesh1rec[:, 0], mesh1rec[:, 1], mesh1rec[:, 2], mesh1rec[:, 3]
                    )
                    mesh2, contour2, midline2 = u.split_mesh2mesh(
                        mesh2rec[:, 0], mesh2rec[:, 1], mesh2rec[:, 2], mesh2rec[:, 3]
                    )
                except Exception as e:
                    bactoscoop_logger.debug(f"{e}")
                    continue
                if 4 <= len(mesh1) and len(mesh2) <= 800:
                    new_meshes.append(mesh1)
                    new_meshes.append(mesh2)

                    new_contours.append(contour1)
                    new_contours.append(contour2)

                    new_midlines.append(midline1)
                    new_midlines.append(midline2)
                else:
                    bactoscoop_logger.debug("Splitting resulted in a mesh too large")

            else:

                new_meshes.append(complete_mesh)
                new_contours.append(contour)
                new_midlines.append(midline)
        self.processed_mesh_dataframe = pd.DataFrame(
            {"mesh": new_meshes, "contour": new_contours, "midline": new_midlines}
        )
        self.processed_mesh_dataframe["frame"] = self.frame
        self.processed_mesh_dataframe["image_name"] = self.image_name

    def object_detection(
        self,
        channels=None,
        smoothing=0.1,
        align=False,
        log_sigma=3,
        kernel_width=4,
        min_overlap_ratio=0.01,
        max_external_ratio=0.1,
    ):
        """
        Detect subcellular objects within cell contours in specified channels.

        This method performs object detection within the contours of cells for the specified channels. It uses a Laplacian of Gaussian (LoG)
        filter to detect subcellular objects and returns their contours. The detected objects are stored in a DataFrame, including the frame,
        cell ID, channel name, and object contours.

        Parameters
        ----------
        channels : List of str, optional
            The list of channel names for which to perform object detection.

        log_sigma : int, optional
            The sigma parameter of the LoG filter used in object detection. Increasing this parameter may detect larger objects.
            The default is 3.

        kernel_width : int, optional
            The kernel width used for dilating object masks. Increasing this value results in larger masks.
            The default is 4.

        min_overlap_ratio : float, optional
            The minimum overlap ratio between detected objects and cell contours to keep the object. Objects with lower overlap are discarded.
            The default is 0.01.

        max_external_ratio : float, optional
            The maximum ratio an object is allowed to lie outside the cell contour. Objects exceeding this ratio are discarded.
            The default is 0.1.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing information about the detected objects, including frame, cell ID, channel name, and object contours.
        """

        all_objects = []

        for channel in channels:

            if channel in self.channels:

                for cell in self.cells:
                    if align:
                        img = self.image
                    else:
                        img = None
                    try:
                        cell.object_meshdata[channel] = u.get_object_mesh(
                            cell.contour,
                            self.channels[channel],
                            img,
                            cell.cell_id,
                            smoothing,
                            log_sigma,
                            kernel_width,
                            min_overlap_ratio,
                            max_external_ratio,
                        )

                        if cell.object_meshdata[channel]["object_contour"]:
                            all_objects.append(
                                {
                                    "frame": self.frame,
                                    "cell_id": cell.cell_id,
                                    "cell_mesh": cell.mesh,
                                    "channel": channel,
                                    "object_contours": cell.object_meshdata[channel][
                                        "object_contour"
                                    ],
                                    "object_mesh": cell.object_meshdata[channel][
                                        "object_mesh"
                                    ],
                                    "object_midline": cell.object_meshdata[channel][
                                        "object_midline"
                                    ],
                                }
                            )

                    except IndexError as e:
                        bactoscoop_logger.info(f"{e}")

        detections_df = pd.DataFrame(all_objects)

        return detections_df

    def shift_correction(
        self,
        shifted_channel,
        log_sigma,
        kernel_width,
        min_overlap_ratio,
        max_external_ratio,
        phase_log_sigma,
        phase_closing_level,
        signal_closing_level,
        max_shift_correction,
    ):

        for channel in shifted_channel:
            if channel not in self.channels:
                raise ValueError(
                    f"'{shifted_channel}' is not present in the channels attribute."
                )
            failed_attempts = 0
            for cell in self.cells:
                try:
                    cell.shifted_contour, cell.shifted_mesh, cell.shifted_midline = (
                        u.shift_contour(
                            self.image,
                            self.channels[channel],
                            cell.contour,
                            cell.mesh,
                            cell.midline,
                            log_sigma,
                            kernel_width,
                            min_overlap_ratio,
                            max_external_ratio,
                            phase_log_sigma,
                            phase_closing_level,
                            signal_closing_level,
                            max_shift_correction,
                        )
                    )
                except Exception as e:
                    bactoscoop_logger.debug(f"Exception {e} in cell : {cell.cell_id}")
                    cell.shifted_contour = None
                    cell.shifted_mesh = None
                    cell.shifted_midline = None
                if cell.shifted_contour is None:
                    failed_attempts += 1
        return (len(self.cells) - failed_attempts) / len(self.cells)


class Cell:
    """
    Class representing a single cell in an image.

    This class represents a single cell within an image, characterized by its contour, mesh, and midline data. It is identified by a unique cell ID.

    Attributes
    ----------
    discard : bool
        A flag to indicate whether the cell should be discarded during analysis.

    contour : ndarray
        The contour data of the cell.

    mesh : ndarray
        The mesh data (4xN).

    midline : ndarray
        The midline data describing the cell's central axis (2xN).

    cell_id : int
        A unique identifier for the cell.

    x1, y1, x2, y2 : ndarray
        Arrays representing the coordinates of the cell's mesh.

    profile_mesh : None
        Placeholder for profile mesh data.

    object_contours : dict
        A dictionary to store detected subcellular object contours in different channels.

    morphological_features : dict
        Dictionary to store morphological features of the cell.

    phase_features : dict
        Dictionary to store phase contrast features of the cell.

    objects_features : dict
        Dictionary to store features related to subcellular objects within the cell.

    svm_features : dict
        Dictionary to store support vector machine (SVM) features.

    profiling_features : dict
        Dictionary to store profiling features.

    profiling_data : dict
        Dictionary to store data related to the profiling of the cell.
    """

    def __init__(self, contour, mesh, midline, shape, cell_id):

        self.discard = False
        self.contour = contour
        self.mesh = mesh
        self.midline = midline
        self.cell_id = cell_id

        self.shifted_contour = None
        self.shifted_mesh = None
        self.shifted_midline = None

        self.x1 = self.mesh[:, 0]
        self.y1 = self.mesh[:, 1]
        self.x2 = self.mesh[:, 2]
        self.y2 = self.mesh[:, 3]

        self.profile_mesh = None

        self.object_meshdata = {}

        self.morphological_features = {}
        self.phase_features = {}
        self.objects_features = {}
        self.svm_features = {}
        self.profiling_features = {}
        self.signal_correlation_features = {}
        self.profiling_data = {}
