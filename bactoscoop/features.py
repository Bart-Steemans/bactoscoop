# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:45:56 2023

@author: Bart Steemans. Govers Lab.
"""
from . import utilities as u
import numpy as np
import logging

bactoscoop_logger = logging.getLogger("logger")


class Features:

    def __init__(self, image_obj):

        self.image_obj = image_obj

    def discard_large_meshes(self, cell, max_mesh_size=600):
        """
        Discard cells with meshes that are too small or large.

        This method checks the size of a cell's mesh and flags it for discard if the mesh size does not fall within the acceptable range.

        Parameters
        ----------
        cell : Cell
            The Cell object representing the cell to be checked.

        Returns
        -------
        None
        """

        if not 4 <= len(cell.mesh) <= max_mesh_size:
            cell.discard = True
            bactoscoop_logger.debug(
                f"The mesh of Cell number {cell.cell_id} is too small or large"
            )

    def discard_bad_meshes(self, cell):

        px = self.image_obj.px
        step_length = u.get_step_length(cell.x1, cell.y1, cell.x2, cell.y2, px)
        _, width_not_ordered = u.get_avg_width(cell.x1, cell.y1, cell.x2, cell.y2, px)
        cell_area = u.get_area(cell.contour, px)
        mesh_area = u.get_mesh_area(width_not_ordered, step_length)
        ratio = mesh_area / cell_area

        if 0.9 <= ratio <= 1.1:
            cell.discard = False
        else:
            bactoscoop_logger.debug(f"Cell mesh with cell id {cell.cell_id} is discarded due to bad mesh quality")
            cell.discard = True

    def discard_bad_object_meshes(self, cell, ratio_list):
        for ratio in ratio_list:
            if 0.9 <= ratio <= 1.1:
                cell.discard = False
            else:
                bactoscoop_logger.debug(
                    f"Object Mesh in Cell id {cell.cell_id} is discarded"
                )
                cell.discard = True

    def contour_selection(self, cell, use_shifted_contours):
        if use_shifted_contours and cell.shifted_contour is not None:
            return cell.shifted_contour, cell.shifted_midline, cell.shifted_mesh
        else:
            return cell.contour, cell.midline, cell.mesh

    def morphological(self, cell, channel, all_data, *args):
        """
        Calculate morphological features for a cell.

        """
        cell.morphological_features = {}

        # Prepare cropped mask and image for morphological measurements and constriction degree
        if self.image_obj.inverted_image is None:
            self.image_obj.get_inverted_image()

        cell_mask, cropped_signal, _, _, x_offset, y_offset = u.get_cell_mask(
            cell.contour, cell.midline, self.image_obj.inverted_image
        )
        cropped_contour, cropped_ml, cropped_msh = u.get_cropped_cell_data(
            cell.contour, cell.midline, cell.mesh, x_offset, y_offset
        )
        cropped_signal_interp2d = u.interp2d(cropped_signal)

        # Morphological measurements
        px = self.image_obj.px
        step_length = u.get_step_length(cell.x1, cell.y1, cell.x2, cell.y2, px)

        cell.morphological_features["cell_length"] = u.get_length(step_length)
        cell.morphological_features["cell_width"], width_not_ordered = u.get_avg_width(
            cell.x1, cell.y1, cell.x2, cell.y2, px
        )
        cell.morphological_features["cell_aspect_ratio"] = (
            cell.morphological_features["cell_length"]
            / cell.morphological_features["cell_width"]
        )
        cell.profiling_data["cell_widthno"] = width_not_ordered

        cell_area = u.get_area(cell.contour, px)
        cell.morphological_features["cell_area"] = cell_area
        cell.morphological_features["cell_volume"] = u.get_volume(
            cell.x1, cell.y1, cell.x2, cell.y2, step_length, px
        )
        cell.morphological_features["cell_surface_area"] = u.get_surface_area(
            width_not_ordered, step_length
        )
        cell.morphological_features["cell_SOV"] = u.get_surface_area_over_volume(
            cell.morphological_features["cell_surface_area"],
            cell.morphological_features["cell_volume"],
        )

        per, circ, poa, sph = u.get_cell_perimeter_measurements(
            cell.contour, cell.morphological_features["cell_area"], px
        )
        cell.morphological_features["cell_perimeter"] = per
        cell.morphological_features["cell_circularity"] = circ
        cell.morphological_features["cell_compactness"] = poa
        cell.morphological_features["cell_sphericity"] = sph
        cell.morphological_features["cell_width_variability"] = (
            u.get_cell_width_variability(width_not_ordered)
        )

        ecc, sol, conv = u.get_cell_convexity(cell_mask)
        cell.morphological_features["cell_eccentricity"] = ecc
        cell.morphological_features["cell_solidity"] = sol
        cell.morphological_features["cell_convexity"] = conv

        con_curvature, con_max_c, con_min_c, con_mean_c, con_std_c = (
            u.get_curvature_characteristics(cell.contour, px)
        )
        cell.morphological_features["cell_max_contour_curvature"] = con_max_c
        cell.morphological_features["cell_min_contour_curvature"] = con_min_c
        cell.morphological_features["cell_mean_contour_curvature"] = con_mean_c
        cell.morphological_features["cell_std_contour_curvature"] = con_std_c
        cell.morphological_features["cell_contour_bending_energy"] = u.bending_energy(
            cell.contour, con_curvature, px
        )

        mid_curvature, mid_max_c, mid_min_c, mid_mean_c, mid_std_c = (
            u.get_curvature_characteristics(cell.midline, px)
        )
        cell.morphological_features["cell_max_midline_curvature"] = mid_max_c
        cell.morphological_features["cell_min_midline_curvature"] = mid_min_c
        cell.morphological_features["cell_mean_midline_curvature"] = mid_mean_c
        cell.morphological_features["cell_std_midline_curvature"] = mid_std_c
        cell.morphological_features["cell_midline_bending_energy"] = u.bending_energy(
            cell.midline, mid_curvature, px
        )
        cell.morphological_features["cell_midline_sinuosity"] = u.sinuosity(
            cell.midline
        )

        cell.morphological_features["cell_area_asymmetry"] = u.get_area_asymmetry(
            step_length, width_not_ordered
        )

        regionprops_features = u.get_additional_regionprops_features(cell_mask, px)

        # Store additional regionprops features
        cell.morphological_features["max_feret_diameter"] = regionprops_features[
            "F_MAX_FERET_DIAMETER"
        ]
        cell.morphological_features["equivalent_diameter"] = regionprops_features[
            "F_EQUIVALENT_DIAMETER"
        ]
        cell.morphological_features["max_radius"] = regionprops_features[
            "F_MAXIMUM_RADIUS"
        ]
        cell.morphological_features["mean_radius"] = regionprops_features[
            "F_MEAN_RADIUS"
        ]
        cell.morphological_features["median_radius"] = regionprops_features[
            "F_MEDIAN_RADIUS"
        ]
        cell.morphological_features["extent"] = regionprops_features["F_EXTENT"]

        # Store moment-related features
        moment_types = [
            "moments",
            "moments_central",
            "moments_normalized",
            "moments_hu",
            "inertia_tensor",
            "inertia_tensor_eigvals",
        ]
        for moment_type in moment_types:
            for key in regionprops_features:
                if f"{moment_type}" in key:
                    cell.morphological_features[key] = regionprops_features[key]

        # Constriction degree calculations
        width_px = u.get_width(cell.x1, cell.y1, cell.x2, cell.y2) * px

        width_profile = width_px
        (
            cell.morphological_features["cell_constriction_degree_width"],
            cell.morphological_features["cell_rel_constr_pos_width"],
            constrDegree_abs,
            ctpos,
        ) = u.constr_degree_single_cell_min(width_profile, step_length)

        axial_profile = u.measure_smoothened_intensity(
            cropped_ml, cropped_signal_interp2d, width=5
        )
        (
            cell.morphological_features["cell_constriction_degree_axial"],
            cell.morphological_features["cell_rel_constr_pos_axial"],
            _,
            _,
        ) = u.constr_degree_single_cell_min(axial_profile, step_length)

        weighted_axial_profile = u.get_weighted_intprofile(axial_profile, width_px)
        (
            cell.morphological_features["cell_constriction_degree_width_axial"],
            cell.morphological_features["cell_rel_constr_pos_width_axial"],
            _,
            _,
        ) = u.constr_degree_single_cell_min(weighted_axial_profile, step_length)

        try:
            cell.profile_mesh = u.get_profile_mesh(
                cropped_msh, u.get_width(cell.x1, cell.y1, cell.x2, cell.y2)
            )
        except MemoryError:
            cell.discard = True
            pass
        mesh_intensity = cropped_signal_interp2d.ev(
            cell.profile_mesh[0], cell.profile_mesh[1]
        ).T
        mesh_intensity_profile = u.gaussian_smoothing(mesh_intensity)
        (
            cell.morphological_features["cell_constriction_degree_mesh"],
            cell.morphological_features["cell_rel_constr_pos_mesh"],
            _,
            _,
        ) = u.constr_degree_single_cell_min(mesh_intensity_profile, step_length)

        # Discarding cells with an erroneous mesh
        self.discard_bad_meshes(cell)

    def profiling(
        self, cell, channel, all_data, use_shifted_contours, shift_signal, *args
    ):
        """
        Calculate profiling features for a cell.

        """
        cell.profiling_features = {}
        px = self.image_obj.px

        if self.image_obj.channels is None or channel not in self.image_obj.channels:
            raise AttributeError(
                f"No objects are detected for channel {channel} or the channel is not loaded"
            )
            return

        if channel not in self.image_obj.bg_channels:
            self.image_obj.get_bg_channel(channel)
            bactoscoop_logger.debug("Background subtraction completed")

        contour, midline, mesh = self.contour_selection(cell, use_shifted_contours)

        if shift_signal:
            cropped_signal, cropped_mask, _, x_offset, y_offset = u.crop_image(
                self.image_obj.bg_channels[channel],
                contour,
                phase_img=self.image_obj.image,
            )
        else:
            cropped_signal, cropped_mask, _, x_offset, y_offset = u.crop_image(
                self.image_obj.bg_channels[channel], contour, phase_img=None
            )

        cropped_contour, cropped_ml, cropped_msh = u.get_cropped_cell_data(
            contour, midline, mesh, x_offset, y_offset
        )
        cropped_signal_interp2d = u.interp2d(cropped_signal)

        try:
            profile_mesh = u.get_profile_mesh(
                cropped_msh, u.get_width(cell.x1, cell.y1, cell.x2, cell.y2)
            )
        except MemoryError:
            cell.discard = True
            pass

        # Intensity features
        cell.profiling_features["axial_intensity"] = u.measure_smoothened_intensity(
            cropped_ml, cropped_signal_interp2d, width=10
        )
        cell.profiling_features["step_length_demograph"] = (
            u.get_cumulative_step_length_midline(cropped_ml, px)
        )
        cell.profiling_features["normalized_axial_intensity"] = u.normalize_per_cell(
            cell.profiling_features["axial_intensity"]
        )
        cell.profiling_features["cell_length"] = u.line_length(cropped_ml) * px
        mesh_intensity = cropped_signal_interp2d.ev(profile_mesh[0], profile_mesh[1]).T
        cell.profiling_features["raw_average_mesh_intensity"] = np.average(
            mesh_intensity, axis=0
        )
        cell.profiling_features["average_mesh_intensity"] = u.gaussian_smoothing(
            mesh_intensity
        )
        cell.profiling_features["raw_normalized_average_mesh_intensity"] = (
            u.normalize_per_cell(cell.profiling_features["raw_average_mesh_intensity"])
        )
        cell.profiling_features["normalized_average_mesh_intensity"] = (
            u.normalize_per_cell(cell.profiling_features["average_mesh_intensity"])
        )

        cell.profiling_features["normalized_average_mesh_intensity"] = (
            u.normalize_per_cell(cell.profiling_features["average_mesh_intensity"])
        )

        step_length = u.get_step_length(cell.x1, cell.y1, cell.x2, cell.y2, px)
        # useful for DAPI signal
        (
            cell.profiling_features["signal_constriction_degree_mesh"],
            cell.profiling_features["signal_rel_constr_pos_mesh"],
            _,
            _,
        ) = u.constr_degree_single_cell_min(
            cell.profiling_features["average_mesh_intensity"], step_length
        )

        (
            glcm_dissimilarity,
            glcm_correlation,
            glcm_homogeneity,
            glcm_energy,
            glcm_contrast,
            glcm_entropy,
        ) = u.calculate_haralick_features(
            cropped_contour,
            cropped_signal,
        )
        cell.profiling_features["cell_mesh_intensity_mean"] = np.mean(
            cell.profiling_features["average_mesh_intensity"]
        )
        cell.profiling_features["cell_mesh_intensity_std"] = np.std(
            cell.profiling_features["average_mesh_intensity"]
        )
        cell.profiling_features["cell_mesh_kurtosis"] = u.get_kurtosis(
            cell.profiling_features["average_mesh_intensity"]
        )
        cell.profiling_features["cell_mesh_skewness"] = u.get_skew(
            cell.profiling_features["average_mesh_intensity"]
        )    
        
            
            
        cell.profiling_features["cell_glcm_dissimilarity"] = glcm_dissimilarity
        cell.profiling_features["cell_glcm_correlation"] = glcm_correlation
        cell.profiling_features["cell_glcm_homogeneity"] = glcm_homogeneity
        cell.profiling_features["cell_glcm_energy"] = glcm_energy
        cell.profiling_features["cell_glcm_contrast"] = glcm_contrast
        cell.profiling_features["cell_glcm_shannon_entropy"] = glcm_entropy

        avg_width, _ = u.get_avg_width_no_px(cell.x1, cell.y1, cell.x2, cell.y2)
        cell.profiling_features["radial_intensity_distribution"] = u.radial_intensities(
            cropped_signal_interp2d,
            cropped_contour,
            width=avg_width,
            min_val=np.min(cropped_signal),
            max_val=np.max(cropped_signal),
            num_erosions=10,
            erosion_scale=None,
        )

        # Discarding cells with an erroneous mesh
        self.discard_bad_meshes(cell)

    def svm(self, cell, channel, all_data, *args):
        """
        Calculate features of a cell to be used in training the Support Vector Machine (SVM) or curate a dataset using a trained SVM model.

        """
        cell.svm_features = {}

        # Preparation cropped mask and images for intensity measurements
        if self.image_obj.inverted_image is None:
            self.image_obj.get_inverted_image()

        cropped_signal, cropped_mask, cropped_contour, x_offset, y_offset = (
            u.crop_image(self.image_obj.inverted_image, cell.contour)
        )
        cropped_signal_interp2d = u.interp2d(cropped_signal)

        px = self.image_obj.px
        step_length = u.get_step_length(cell.x1, cell.y1, cell.x2, cell.y2, px)

        # Contour is added here for labeling purposes
        cell.svm_features["contour"] = cell.contour

        # Morphological features
        cell.svm_features["cell_length"] = u.get_length(step_length)
        cell.svm_features["cell_width"], width_not_ordered = u.get_avg_width(
            cell.x1, cell.y1, cell.x2, cell.y2, px
        )
        cell.svm_features["cell_area"] = u.get_area(cell.contour, px)
        cell.svm_features["cell_volume"] = u.get_volume(
            cell.x1, cell.y1, cell.x2, cell.y2, step_length, px
        )
        cell.svm_features["cell_surface_area"] = u.get_surface_area(
            width_not_ordered, step_length
        )
        (
            curvature,
            cell.svm_features["max_curvature"],
            cell.svm_features["min_curvature"],
            cell.svm_features["mean_curvature"],
            std_c,
        ) = u.get_curvature_characteristics(cell.contour, px)
        (
            cell.svm_features["cell_perimeter"],
            cell.svm_features["cell_circularity"],
            cell.svm_features["cell_POA"],
            cell.svm_features["cell_sphericity"],
        ) = u.get_cell_perimeter_measurements(
            cell.contour, cell.svm_features["cell_area"], px
        )

        # Cell surface intensity features
        total_int, max_int, mean_int = u.get_total_phaco_intensity(
            cropped_contour, cropped_signal.shape, cropped_signal_interp2d
        )
        cell.svm_features["phaco_total_intensity"] = total_int
        cell.svm_features["phaco_max_intensity"] = max_int
        cell.svm_features["phaco_mean_intensity"] = mean_int

        # Contour intensity features
        phaco_contour_intensity = u.get_contour_intensity(
            cropped_contour, cropped_signal_interp2d
        )
        phaco_contour_intensity_variability = u.measure_contour_variability(
            phaco_contour_intensity
        )

        sorted_phaco_contour_intensity = sorted(phaco_contour_intensity)
        cell.svm_features["phaco_contour_peaks"] = u.find_signal_peaks(
            phaco_contour_intensity, max_int
        )
        cell.svm_features["phaco_max_contour_intensity"] = np.max(
            phaco_contour_intensity
        )
        cell.svm_features["phaco_mean_contour_intensity"] = np.mean(
            phaco_contour_intensity
        )
        cell.svm_features["phaco_min_contour_intensity"] = np.mean(
            sorted_phaco_contour_intensity[:10]
        )
        cell.svm_features["phaco_max_contour_variability"] = np.max(
            phaco_contour_intensity_variability
        )
        cell.svm_features["phaco_mean_contour_variability"] = np.mean(
            phaco_contour_intensity_variability
        )

        # Midline intensity features
        # If no midline available calculate it with mesh2midline( ).
        cropped_midline = cell.midline.copy()
        cropped_midline -= [x_offset, y_offset]
        cell.profiling_data["phaco_axial_intensity"] = u.measure_smoothened_intensity(
            cropped_midline, cropped_signal_interp2d, width=5
        )

        cell.svm_features["midline_kurtosis"] = u.get_kurtosis(
            cell.profiling_data["phaco_axial_intensity"]
        )
        cell.svm_features["midline_skewness"] = u.get_skew(
            cell.profiling_data["phaco_axial_intensity"]
        )

        # Expanded contour intensity features
        expanded_contour = u.expand_contour(cropped_contour, scale=2)
        eroded_contour = u.erode_contour(cropped_contour, scale=2)
        cell.profiling_data["phaco_expanded_contour_intensity"] = (
            u.get_contour_intensity(expanded_contour, cropped_signal_interp2d)
        )
        cell.profiling_data["phaco_eroded_contour_intensity"] = u.get_contour_intensity(
            eroded_contour, cropped_signal_interp2d
        )

        cell.svm_features["phaco_max_expanded_contour_intensity"] = np.max(
            cell.profiling_data["phaco_expanded_contour_intensity"]
        )
        cell.svm_features["phaco_mean_expanded_contour_intensity"] = np.mean(
            cell.profiling_data["phaco_expanded_contour_intensity"]
        )

        # Mesh gradient features
        cell.svm_features["phaco_cell_edge_gradient"] = np.average(
            cell.profiling_data["phaco_eroded_contour_intensity"]
            - cell.profiling_data["phaco_expanded_contour_intensity"]
        )

        # Discarding cells with an erroneous mesh
        self.discard_bad_meshes(cell)

    def objects(self, cell, channel, all_data, use_shifted_contours, *args):
        """
        Calculates features related to the identified objects within a cell.

        """
        cell.objects_features = {}

        px = self.image_obj.px

        if self.image_obj.channels is None or channel not in self.image_obj.channels:
            raise AttributeError(
                f"No objects are detected for channel {channel} or the channel is not loaded"
            )
            return

        if channel not in self.image_obj.bg_channels:
            self.image_obj.get_bg_channel(channel)
            bactoscoop_logger.debug("Background subtraction completed")

        if cell.object_meshdata[channel]["object_contour"]:

            # Extract required cellular/object properties to extract the features
            step_length = u.get_step_length(cell.x1, cell.y1, cell.x2, cell.y2, px)
            cell_volume = u.get_volume(
                cell.x1, cell.y1, cell.x2, cell.y2, step_length, px
            )
            cell_area = u.get_area(cell.contour, px)
            cell_widthno = u.get_width(cell.x1, cell.y1, cell.x2, cell.y2)
            # Object properties and cropping to speed up analysis
            object_contours = cell.object_meshdata[channel]["object_contour"]
            object_meshes = cell.object_meshdata[channel]["object_mesh"]
            object_midlines = cell.object_meshdata[channel]["object_midline"]
            (
                object_masks,
                cropped_signal,
                cropped_oc,
                cropped_oml,
                cropped_oms,
                x_offset,
                y_offset,
            ) = u.get_object_masks(
                cell.contour,
                object_contours,
                object_midlines,
                object_meshes,
                self.image_obj.bg_channels[channel],
            )

            cropped_contour, cropped_ml, cropped_msh = u.get_cropped_cell_data(
                cell.contour, cell.midline, cell.mesh, x_offset, y_offset
            )
            cropped_signal_interp2d = u.interp2d(cropped_signal)

            # Calculations without object mesh
            (
                rectl,
                rectw,
                cell.objects_features["cell_total_obj_rect_length"],
                cell.objects_features["cell_avg_obj_rect_width"],
            ) = u.get_object_rect_length(object_contours, px)
            aspect_r, cell.objects_features["cell_avg_obj_aspectratio"] = (
                u.get_object_aspect_ratio(rectl, rectw)
            )
            cell.objects_features["cell_avg_obj_width_dt"], width_no_dt = (
                u.get_object_avg_width(object_masks, px)
            )
            cell.objects_features["object_number"] = len(object_contours)

            # Object area and NC ratio
            area_list, total_obj_area = u.get_object_area(object_contours, px)
            cell.objects_features["cell_total_obj_area"] = total_obj_area
            cell.objects_features["NC_ratio"] = total_obj_area / cell_area

            # Curvature characteristics
            (
                curvatures,
                max_c,
                min_c,
                mean_c,
                std_c,
                cell.objects_features["cell_avg_obj_mean_c"],
                cell.objects_features["cell_avg_obj_std_c"],
            ) = u.get_object_curvature_characteristics(object_contours, px)

            # Perimeter characteristics
            (
                perimeters,
                circularities,
                compactnesses,
                sphericities,
                avg_peri,
                avg_cir,
                avg_comp,
                avg_sph,
            ) = u.get_object_perimeter_measurements(object_contours, area_list, px)
            cell.objects_features["cell_avg_obj_perimeter"] = avg_peri
            cell.objects_features["cell_avg_obj_circularity"] = avg_cir
            cell.objects_features["cell_avg_obj_compactness"] = avg_comp
            cell.objects_features["cell_avg_obj_sphericity"] = avg_sph

            (
                convexity,
                eccentricity,
                solidity,
                avg_convexity,
                avg_eccentricity,
                avg_solidity,
            ) = u.get_object_convexity(object_masks)
            cell.objects_features["cell_avg_obj_convexity"] = avg_convexity
            cell.objects_features["cell_avg_obj_eccentricity"] = avg_eccentricity
            cell.objects_features["cell_avg_obj_solidity"] = avg_solidity

            # Object volume using approximation method (https://doi.org/10.1016/j.cell.2021.05.037)
            object_volumes, total_volume = u.get_approx_nucleoid_volume(
                cell_volume, area_list, cell_area
            )
            cell.objects_features["cell_total_obj_volume_approx"] = total_volume

            # Calculations using object mesh
            obj_x1, obj_y1, obj_x2, obj_y2 = u.get_obj_mesh_coords(cropped_oms)
            axial_intensity = u.measure_along_objects_midline_interp2d(
                cropped_oml, cropped_signal_interp2d, width=5
            )
            obj_step_lengths, obj_lengths = u.get_object_step_length(
                obj_x1, obj_y1, obj_x2, obj_y2, px
            )
            cell.objects_features["cell_total_obj_mesh_length"] = sum(obj_lengths)
            width_list, width_no, cell.objects_features["cell_avg_obj_width"] = (
                u.get_obj_avg_width(obj_x1, obj_y1, obj_x2, obj_y2, px)
            )
            width_var_list, cell.objects_features["cell_avg_obj_width_variability"] = (
                u.get_object_width_variability(width_no)
            )

            (
                constrDegree,
                relPos,
                absconstrDegree,
                ctpos,
                cell.objects_features["object_avg_consDegree"],
                avg_abs_consDegree,
            ) = u.constr_degree(axial_intensity, obj_step_lengths, width_no)
            object_volumes, cell.objects_features["cell_total_obj_volume"] = (
                u.get_nucleoid_volume(
                    obj_x1, obj_y1, obj_x2, obj_y2, obj_step_lengths, px
                )
            )
            surface_area_list, cell.objects_features["obj_surface_area"] = (
                u.get_object_surface_area(width_no, obj_step_lengths)
            )

            sov_list, cell_avg_sov = u.get_object_surface_area_over_volume(
                surface_area_list, object_volumes
            )
            cell.objects_features["cell_avg_obj_SOV"] = cell_avg_sov

            (
                contour_bending_energies,
                cell.objects_features["cell_obj_contour_avg_bending_energy"],
            ) = u.object_bending_energy(object_contours, px)
            (
                midline_bending_energies,
                cell.objects_features["cell_obj_midline_avg_bending_energy"],
            ) = u.object_bending_energy(object_midlines, px)
            sinuosity_list, cell.objects_features["cell_avg_midline_sinuosity"] = (
                u.get_object_sinuosity(object_midlines)
            )

            object_centroid_list = u.get_object_centroid(object_masks)
            (
                cell.objects_features["cell_avg_interobject_distance"],
                interobject_distances,
            ) = u.get_interobject_distance(object_centroid_list, px)

            pole1, pole2 = u.get_pole_coordinates(cropped_msh)

            (
                cell.objects_features["avg_pole_obj_distance"],
                min2distances,
                pole_distances,
            ) = u.get_pole_object_distances(cropped_msh, object_centroid_list, px)
            (
                cell.objects_features["l"],
                cell.objects_features["d"],
                cell.objects_features["l_norm"],
                cell.objects_features["d_norm"],
            ) = u.projection_on_midline(
                object_centroid_list, cropped_ml, cell_widthno, px
            )
            cell.objects_features["object_avg_center_distance"] = (
                u.get_avg_distance_from_center(cell.objects_features["l_norm"])
            )

            # Texture features
            (
                glcm_object_dissimilarity,
                glcm_object_correlation,
                glcm_object_homogeneity,
                glcm_object_energy,
                glcm_object_contrast,
                glcm_object_entropy,
            ) = u.calculate_object_haralick_features(object_masks, cropped_signal)
            cell.objects_features["glcm_obj_dissimilarity"] = glcm_object_dissimilarity
            cell.objects_features["glcm_obj_correlation"] = glcm_object_correlation
            cell.objects_features["glcm_obj_homogeneity"] = glcm_object_homogeneity
            cell.objects_features["glcm_obj_energy"] = glcm_object_energy
            cell.objects_features["glcm_obj_contrast"] = glcm_object_contrast
            cell.objects_features["glcm_obj_shannon_entropy"] = glcm_object_entropy

            object_intensity_values, object_intensities_list = u.get_object_intensities(
                object_masks, cropped_signal
            )
            cell.objects_features["objects_intensity_mean"] = np.mean(
                object_intensity_values
            )
            cell.objects_features["objects_intensity_std"] = np.std(
                object_intensity_values
            )
            cell.objects_features["objects_kurtosis"] = u.get_kurtosis(
                object_intensity_values
            )
            cell.objects_features["objects_skewness"] = u.get_skew(
                object_intensity_values
            )

            if all_data:
                cell.objects_features["obj_areas"] = area_list
                cell.objects_features["obj_aspect_ratios"] = aspect_r
                cell.objects_features["max_obj_curvatures"] = max_c
                cell.objects_features["min_obj_curvatures"] = min_c
                cell.objects_features["mean_obj_curvatures"] = mean_c
                cell.objects_features["std_obj_curvatures"] = std_c
                cell.objects_features["obj_perimeters"] = perimeters
                cell.objects_features["obj_circularities"] = circularities
                cell.objects_features["obj_compactnesses"] = compactnesses
                cell.objects_features["obj_sphericities"] = sphericities
                cell.objects_features["obj_convexities"] = convexity
                cell.objects_features["obj_eccentricities"] = eccentricity
                cell.objects_features["obj_solidities"] = solidity
                cell.objects_features["obj_volumes_approx"] = object_volumes
                cell.objects_features["obj_SOV"] = sov_list
                cell.objects_features["obj_constriction_degrees"] = constrDegree
                cell.objects_features["obj_relative_positions"] = relPos
                cell.objects_features["obj_contour_bending_energies"] = (
                    contour_bending_energies
                )
                cell.objects_features["obj_midline_bending_energies"] = (
                    contour_bending_energies
                )
                cell.objects_features["obj_interobject_distances"] = (
                    interobject_distances
                )
                cell.objects_features["obj_midline_sinuosities"] = sinuosity_list
                cell.objects_features["min_2_distances_from_pole"] = min2distances
                cell.objects_features["obj_pole_distances"] = pole_distances
                cell.objects_features["obj_intensity_values"] = object_intensity_values
                cell.objects_features["obj_mean_intensities"] = u.get_objects_mean(
                    object_intensities_list
                )
                cell.objects_features["obj_std_intensities"] = u.get_objects_std(
                    object_intensities_list
                )
                cell.objects_features["obj_kurtosis_intensities"] = (
                    u.get_objects_kurtosis(object_intensities_list)
                )
                cell.objects_features["obj_skewness_intensities"] = (
                    u.get_objects_skewness(object_intensities_list)
                )

            # Discard bad object mesh based on mesh area - contour area ratio
            ratio_list = u.ratio_mesh_over_total_area(
                obj_step_lengths, width_no, area_list
            )
            self.discard_bad_object_meshes(cell, ratio_list)
        else:
            raise ValueError("No object contours to calculate features from")
            cell.discard = True

    def membrane(self, cell, channel, all_data, use_shifted_contours, shift_signal):
        cell.membrane_features = {}
        px = self.image_obj.px

        if self.image_obj.channels is None or channel not in self.image_obj.channels:
            raise AttributeError(
                f"No objects are detected for channel {channel} or the channel is not loaded"
            )
            return

        if channel not in self.image_obj.bg_channels:
            self.image_obj.get_bg_channel(channel)
            bactoscoop_logger.debug("Background subtraction completed")
            
        if cell.object_meshdata[channel]["object_contour"]:
            
            contour, midline, mesh = self.contour_selection(cell, use_shifted_contours)

            if shift_signal:
                cropped_signal, cropped_mask, _, x_offset, y_offset = u.crop_image(
                    self.image_obj.bg_channels[channel],
                    contour,
                    phase_img=self.image_obj.image,
                )
            else:
                cropped_signal, cropped_mask, _, x_offset, y_offset = u.crop_image(
                    self.image_obj.bg_channels[channel], 
                    contour, 
                    phase_img=None
                )
    
            cropped_contour, _, cropped_msh = u.get_cropped_cell_data(
                contour, midline, mesh, x_offset, y_offset
            )
            cropped_signal_interp2d = u.interp2d(cropped_signal)
    
            p2p_contour = u.mesh2contour(
                cropped_msh[:, 0], cropped_msh[:, 1], cropped_msh[:, 2], cropped_msh[:, 3]
            )
            eroded_contour = u.erode_contour(p2p_contour, scale=2)
            area = u.get_area(cropped_contour, px)
            cell.membrane_features["cell_perimeter"], _, _, _ = (
                u.get_cell_perimeter_measurements(cropped_contour, area, px)
            )
            binned_eroded_contour = u.change_contour_array_length(
                cell.membrane_features["cell_perimeter"], eroded_contour
            )
            contour_intensities = u.measure_intensity_interp2d(
                binned_eroded_contour, cropped_signal_interp2d, width=6
            )
            contour_intensity_profile = u.gaussian_smoothing(contour_intensities)
            cell.membrane_features["contour_intensity"] = contour_intensity_profile
            cell.membrane_features["normalized_contour_intensity"] = u.normalize_per_cell(
                cell.membrane_features["contour_intensity"]
            )
            cell.membrane_features["complemented_contour_intensity"] = (
                u.get_complemented_contour_intensity(
                    cell.membrane_features["normalized_contour_intensity"]
                )
            )
    
            cell.membrane_features["contour_intensity_mean"] = np.mean(
                contour_intensity_profile
            )
            cell.membrane_features["contour_intensity_std"] = np.std(
                contour_intensity_profile
            )
            cell.membrane_features["contour_intensity_kurtosis"] = u.get_kurtosis(
                contour_intensity_profile
            )
            cell.membrane_features["contour_intensity_skewness"] = u.get_skew(
                contour_intensity_profile
            )
            cell.membrane_features["contour_gradient_homogeneity"] = u.get_homogeneity(
                cell.membrane_features["normalized_contour_intensity"]
            )
    
            (
                glcm_contour_dissimilarity,
                glcm_contour_correlation,
                glcm_contour_homogeneity,
                glcm_contour_energy,
                glcm_contour_contrast,
                glcm_contour_entropy,
            ) = u.calculate_contour_haralick_features(contour_intensities)
            cell.membrane_features["glcm_contour_dissimilarity"] = (
                glcm_contour_dissimilarity
            )
            cell.membrane_features["glcm_contour_correlation"] = glcm_contour_correlation
            cell.membrane_features["glcm_contour_homogeneity"] = glcm_contour_homogeneity
            cell.membrane_features["glcm_contour_energy"] = glcm_contour_energy
            cell.membrane_features["glcm_contour_contrast"] = glcm_contour_contrast
            cell.membrane_features["glcm_contour_shannon_entropy"] = glcm_contour_entropy
    
            self.discard_bad_meshes(cell)
        else:
            raise ValueError("No object contours to calculate features from")
            cell.discard = True
#not yet functional 
    def colocalization(
        self,
        cell,
        signal1,
        signal2,
        shift_signal1,
        shift_signal2,
        all_data,
        use_shifted_contours,
    ):

        cell.signal_correlation_features = {}
        px = self.image_obj.px

        if (
            self.image_obj.channels is None
            or signal1 not in self.image_obj.channels
            or signal2 not in self.image_obj.channels
        ):
            raise AttributeError(
                f"Channel {signal1} or {signal2} is not detected or not loaded"
            )

        if signal1 not in self.image_obj.bg_channels:
            self.image_obj.get_bg_channel(signal1)
            bactoscoop_logger.debug("Background subtraction completed for signal1")

        if signal2 not in self.image_obj.bg_channels:
            self.image_obj.get_bg_channel(signal2)
            bactoscoop_logger.debug("Background subtraction completed for signal2")

        contour, midline, mesh = self.contour_selection(cell, use_shifted_contours)

        cropped_signal1, cropped_mask1, _, x_offset1, y_offset1 = (
            u.crop_signal_with_optional_shift(
                self.image_obj, signal1, contour, shift_signal1
            )
        )
        cropped_signal2, cropped_mask2, _, x_offset2, y_offset2 = (
            u.crop_signal_with_optional_shift(
                self.image_obj, signal2, contour, shift_signal2
            )
        )
