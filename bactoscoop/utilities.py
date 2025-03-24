# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:46:15 2023

@author: Bart Steemans. Govers Lab. 
"""
import os
import shutil
import tifffile
import logging

import numpy as np
import cv2 as cv
import networkx as nx
import math
from shapely import Polygon
from tqdm import tqdm
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

from scipy import interpolate, ndimage, spatial
from scipy.ndimage import gaussian_filter1d, fourier_shift, gaussian_laplace
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.draw import polygon2mask
from skimage.morphology import skeletonize, closing
from skimage.measure import regionprops_table, regionprops, label, shannon_entropy
from skimage.registration import phase_cross_correlation
from scipy.optimize import curve_fit

bactoscoop_logger = logging.getLogger("logger")


# -------------------------------------------------------------------------------------------------------
# FUNCTIONS
# %% Cellular mesh creation functions
def smallest_path(x_coords, y_coords, start_x, start_y):
    G = nx.Graph()
    for i in range(len(x_coords)):
        G.add_node(i, x=x_coords[i], y=y_coords[i])
    for i in range(len(x_coords)):
        for j in range(i + 1, len(x_coords)):
            G.add_edge(
                i,
                j,
                weight=np.sqrt(
                    (x_coords[i] - x_coords[j]) ** 2 + (y_coords[i] - y_coords[j]) ** 2
                ),
            )
    start = None
    for i in range(len(x_coords)):
        if x_coords[i] == start_x and y_coords[i] == start_y:
            start = i
            break
    T = nx.minimum_spanning_tree(G)
    path = nx.dfs_postorder_nodes(T, source=start)
    sorted_points = list(path)
    return [x_coords[i] for i in sorted_points], [y_coords[i] for i in sorted_points]


def skeleton_endpoints(skel):

    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # Apply the convolution.
    kernel = np.uint8([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    filtered = cv.filter2D(skel, -1, kernel)

    out = np.zeros_like(skel)
    out[np.where(filtered == 11)] = 1
    return out


def get_ordered_list(points, x, y):
    distance = ((x - points[1]) ** 2 + (y - points[0]) ** 2) ** 0.5
    args = np.argsort(distance)
    return args


# @jit(nopython=True, cache=True)
# MOMIA
def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):  # Ax+By = C
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    intersect_x = (B1 * C2 - B2 * C1) / (A2 * B1 - A1 * B2)
    intersect_y = (A1 * C2 - A2 * C1) / (B2 * A1 - B1 * A2)
    return intersect_x, intersect_y


# MOMIA
def line_contour_intersection(p1, p2, contour):
    v1, v2 = contour[:-1], contour[1:]
    x1, y1 = v1.T
    x2, y2 = v2.T
    x3, y3 = p1
    x4, y4 = p2
    xy = np.array(line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)).T
    dxy_v1 = xy - v1
    dxy_v2 = xy - v2
    dxy = dxy_v1 * dxy_v2
    intersection_points = xy[np.where(np.logical_and(dxy[:, 0] < 0, dxy[:, 1] < 0))]
    if len(intersection_points) > 2:
        dist = np.sum(
            np.square(np.tile(p1, (len(intersection_points), 1)) - intersection_points),
            axis=1,
        )
        intersection_points = intersection_points[np.argsort(dist)[0:2]]
    return intersection_points


# MOMIA
def unit_perpendicular_vector(data, closed=True):

    p1 = data[1:]
    p2 = data[:-1]
    dxy = p1 - p2
    ang = np.arctan2(dxy.T[1], dxy.T[0]) + 0.5 * np.pi
    dx, dy = np.cos(ang), np.sin(ang)
    unit_dxy = np.array([dx, dy]).T
    if not closed:
        unit_dxy = np.concatenate([[unit_dxy[0]], unit_dxy])
    else:
        unit_dxy = np.concatenate([unit_dxy, [unit_dxy[-1]]])
    return unit_dxy


def get_major_axis_line_coordinates(masks):
    # Find contours in the mask
    contours, _ = cv.findContours(
        masks.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    ellipse = cv.fitEllipse(contours[0])

    # Extract major axis parameters
    # Extract major axis parameters
    center, (major_axis, minor_axis), angle = ellipse

    # Calculate endpoints of the major axis with reversed orientation
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Center coordinates of the major axis
    mid_x = center[0]
    mid_y = center[1]

    # Endpoint coordinates of the major axis with reversed orientation
    endpoint1_x = int(mid_x + 0.5 * major_axis * sin_angle)
    endpoint1_y = int(mid_y - 0.5 * major_axis * cos_angle)
    endpoint2_x = int(mid_x - 0.5 * major_axis * sin_angle)
    endpoint2_y = int(mid_y + 0.5 * major_axis * cos_angle)

    # Generate coordinates along the major axis line
    line_x = np.linspace(endpoint1_x, endpoint2_x, num=200)
    line_y = np.linspace(endpoint1_y, endpoint2_y, num=200)

    # Combine x and y coordinates into an array of shape (n, 2) with (x, y) convention
    line_coordinates = np.column_stack((line_y, line_x))
    return line_coordinates


def extract_skeleton(masks):
    skeleton = skeletonize(masks, method="lee")

    # Retrieve the skeleton coordinates in the original image space
    skeleton_coords = np.column_stack(np.where(skeleton)).astype(int)

    x_skel = skeleton_coords[:, 0]
    y_skel = skeleton_coords[:, 1]

    skel_mask = np.zeros_like(masks)
    skel_mask[x_skel, y_skel] = 1

    # Sort coordinates based on distance
    skel_end = skeleton_endpoints(skel_mask)
    try:

        end_coords = np.argwhere(skel_end == 1)[0]

    except IndexError:

        return get_major_axis_line_coordinates(masks)

    sorted_skelx, sorted_skely = smallest_path(
        x_skel, y_skel, end_coords[0], end_coords[1]
    )

    if len(y_skel) <= 3:
        return get_major_axis_line_coordinates(masks)

    elif 4 <= len(y_skel) <= 13:

        fkc, u = interpolate.splprep([sorted_skelx, sorted_skely], k=1, s=1, per=0)
    else:
        fkc, u = interpolate.splprep([sorted_skelx, sorted_skely], k=2, s=10, per=0)
    smoothed_skel = np.asarray(
        interpolate.splev(np.linspace(u.min(), u.max(), 200), fkc, der=0)
    ).T
    return smoothed_skel


# MOMIA
def find_poles(smoothed_skeleton, smoothed_contour, find_pole1=True, find_pole2=True):
    # find endpoints and their nearest neighbors on a midline
    length = len(smoothed_skeleton)
    extended_pole1 = [smoothed_skeleton[0]]
    extended_pole2 = [smoothed_skeleton[-1]]
    i = 0
    j = 0
    if find_pole1:
        for i in range(10):
            p1 = smoothed_skeleton[i]
            p2 = smoothed_skeleton[i + 1]
            # find the two intersection points between
            # the vectorized contour and line through pole1
            intersection_points_pole1 = line_contour_intersection(
                p1, p2, smoothed_contour
            )
            # find the interesection point with the same direction as the outward pole vector
            dxy_1 = p1 - p2
            p1_tile = np.tile(p1, (len(intersection_points_pole1), 1))
            p1dot = (intersection_points_pole1 - p1_tile) * dxy_1
            index_1 = np.where((p1dot[:, 0] > 0) & (p1dot[:, 1] > 0))[0]
            if len(index_1) > 0:
                extended_pole1 = intersection_points_pole1[index_1]
                break
    else:
        i = 1

    if find_pole2:
        for j in range(10):
            p3 = smoothed_skeleton[-1 - j]
            p4 = smoothed_skeleton[-2 - j]
            # find the two intersection points between
            # the vectorized contour and line through pole1
            intersection_points_pole2 = line_contour_intersection(
                p3, p4, smoothed_contour
            )
            # find the interesection point with the same direction as the outward pole vector
            dxy_2 = p3 - p4
            p3_tile = np.tile(p3, (len(intersection_points_pole2), 1))
            p3dot = (intersection_points_pole2 - p3_tile) * dxy_2
            index_2 = np.where((p3dot[:, 0] > 0) & (p3dot[:, 1] > 0))[0]
            if len(index_2) > 0:
                extended_pole2 = intersection_points_pole2[index_2]
                break
    else:
        j = 1
    trimmed_midline = smoothed_skeleton[i : length - j]
    return extended_pole1, extended_pole2, trimmed_midline


# MOMIA
def extend_skeleton(
    smoothed_skeleton, smoothed_contour, find_pole1=True, find_pole2=True
):
    # initiate approximated tip points
    new_pole1, new_pole2, trimmed_skeleton = find_poles(
        smoothed_skeleton,
        smoothed_contour,
        find_pole1=find_pole1,
        find_pole2=find_pole2,
    )
    extended_skeleton = np.concatenate([new_pole1, trimmed_skeleton, new_pole2])

    extended_skeleton = spline_approximation(
        extended_skeleton, n=200, smooth_factor=1, closed=False
    )
    return extended_skeleton, new_pole1, new_pole2


# MOMIA
def straighten_by_orthogonal_lines(contour, midline, length, width, unit_micron=0.5):
    # estimate profile mesh size
    median_width = np.median(width)
    N_length = int(round(length / unit_micron))
    N_width = int(round(median_width / unit_micron))
    midline = spline_approximation(midline, n=N_length, smooth_factor=1, closed=False)

    half_contour_1, half_contour_2 = divide_contour_by_midline(
        midline, contour
    )  # [:-1]
    # infer orthogonal vectors
    ortho_unit_vectors = unit_perpendicular_vector(midline)
    # generate orthogonal profile lines for all midline points except for the polar ones
    l1 = orthogonal_intersection_point(
        midline, half_contour_1, precomputed_orthogonal_vector=ortho_unit_vectors
    )

    l2 = orthogonal_intersection_point(
        midline, half_contour_2, precomputed_orthogonal_vector=ortho_unit_vectors
    )

    dl = (l2 - l1) / N_width
    mult_mat = np.tile(np.arange(N_width + 1), (len(l1), 1))
    mat_x = l1[:, 0][:, np.newaxis] + mult_mat * dl[:, 0][:, np.newaxis]
    mat_y = l1[:, 1][:, np.newaxis] + mult_mat * dl[:, 1][:, np.newaxis]
    profile_mesh = np.array([mat_x, mat_y])
    return l1, l2, profile_mesh, midline


def radial_intensities(
    cropped_signal_interp2d,
    contour,
    width,
    min_val,
    max_val,
    num_erosions,
    erosion_scale=None,
):

    erosion = width / (2 * num_erosions)
    if erosion_scale is not None:
        erosion = erosion_scale

    eroded_contours = [contour]
    for i in range(num_erosions):
        eroded_contour = erode_contour(eroded_contours[-1], scale=erosion)
        eroded_contours.append(eroded_contour)

    eroded_contours.reverse()

    intensities_per_mesh = []
    for contour in eroded_contours:
        intensities = measure_smoothened_intensity(
            contour, cropped_signal_interp2d, width=erosion, subpixel=0.1
        )
        normalized_intensities = (intensities - min_val) / (max_val - min_val)
        average_intensity = np.average(normalized_intensities)
        intensities_per_mesh.append(average_intensity)

    return intensities_per_mesh


# MOMIA
def spline_approximation(init_contour, n=200, smooth_factor=1, closed=True):
    if closed:
        tck, u = interpolate.splprep(init_contour.T, u=None, s=smooth_factor, per=1)
    else:
        tck, u = interpolate.splprep(init_contour.T, u=None, s=smooth_factor)
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = interpolate.splev(u_new, tck, der=0)
    return np.array([x_new, y_new]).T


# MOMIA
def divide_contour_by_midline(midline, contour):
    dist1 = distance_matrix(contour, midline[0]).flatten()
    dist2 = distance_matrix(contour, midline[-1]).flatten()

    id1, id2 = np.argsort(dist1)[:2]
    id3, id4 = np.argsort(dist2)[:2]

    contour_cp = contour.copy()
    if max(id1, id2) < max(id3, id4):
        term_p1 = max(id1, id2)
        if abs(id3 - id4) == 1:
            term_p2 = max(id3, id4) + 1
        elif abs(id3 - id4) > 1:
            term_p2 = max(id3, id4) + 2
        contour_cp = np.insert(contour_cp, term_p1, midline[0], axis=0)
        contour_cp = np.insert(contour_cp, term_p2, midline[-1], axis=0)

    else:
        term_p1 = max(id3, id4)
        if abs(id1 - id2) == 1:
            term_p2 = max(id1, id2) + 1
        elif abs(id1 - id2) > 1:
            term_p2 = max(id1, id2) + 2
        contour_cp = np.insert(contour_cp, term_p1, midline[-1], axis=0)
        contour_cp = np.insert(contour_cp, term_p2, midline[0], axis=0)

    if term_p1 == term_p2:
        raise ValueError("Two endpoints are identical!")
    else:
        pos1, pos2 = sorted([term_p1, term_p2])
        # print(id1, id2, id3, id4, pos1, pos2, len(contour_cp))
        half_contour_1 = contour_cp[pos1 : min(pos2 + 1, len(contour_cp) - 1)]
        half_contour_2 = np.concatenate([contour_cp[pos2:], contour_cp[: pos1 + 1]])
    return half_contour_1, half_contour_2


# MOMIA
def distance_matrix(data1, data2):
    x1, y1 = data1.T
    x2, y2 = data2.T
    dx = x1[:, np.newaxis] - x2
    dy = y1[:, np.newaxis] - y2
    dxy = np.sqrt(dx**2 + dy**2)
    return dxy


# MOMIA
def orthogonal_intersection_point(
    midline, outerline, precomputed_orthogonal_vector=None, min_dist=1e-100
):
    v1, v2 = outerline[:-1], outerline[1:]
    skel_x, skel_y = midline.T
    if precomputed_orthogonal_vector is None:
        intersect_x, intersect_y = intersect_matrix(midline, outerline)
    else:
        intersect_x, intersect_y = intersect_matrix(
            midline, outerline, orthogonal_vectors=precomputed_orthogonal_vector
        )
    dx_v1 = intersect_x - v1.T[0][:, np.newaxis]
    dx_v2 = intersect_x - v2.T[0][:, np.newaxis]
    dy_v1 = intersect_y - v1.T[1][:, np.newaxis]
    dy_v2 = intersect_y - v2.T[1][:, np.newaxis]
    dx = dx_v1 * dx_v2
    dy = dy_v1 * dy_v2

    dist_x = skel_x[np.newaxis, :] - intersect_x
    dist_y = skel_y[np.newaxis, :] - intersect_y
    # influence on extreme points
    non_bounadry_points = np.where(np.logical_and(dy >= 0, dx >= 0))
    dist_matrix = np.sqrt(dist_x**2 + dist_y**2)
    dist_matrix[non_bounadry_points] = np.inf
    dist_matrix[dist_matrix <= min_dist] = np.inf

    nearest_id_x = np.argsort(dist_matrix, axis=0)[:1]
    nearest_id_y = np.linspace(
        0, dist_matrix.shape[1] - 1, dist_matrix.shape[1]
    ).astype(int)
    pos_list = np.array(
        [
            intersect_x[nearest_id_x[0], nearest_id_y],
            intersect_y[nearest_id_x[0], nearest_id_y],
        ]
    ).T
    return pos_list


# MOMIA
def intersect_matrix(line, contour, orthogonal_vectors=None):
    if orthogonal_vectors is None:
        dxy = unit_perpendicular_vector(line, closed=False)
    else:
        dxy = orthogonal_vectors
    v1, v2 = contour[:-1], contour[1:]
    x1, y1 = v1.T
    x2, y2 = v2.T
    x3, y3 = line.T
    perp_xy = line + dxy
    x4, y4 = perp_xy.T
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    A1B2 = A1[:, np.newaxis] * B2
    A1C2 = A1[:, np.newaxis] * C2
    B1A2 = B1[:, np.newaxis] * A2
    B1C2 = B1[:, np.newaxis] * C2
    C1A2 = C1[:, np.newaxis] * A2
    C1B2 = C1[:, np.newaxis] * B2
    # can give outliers when B1A2 = A1B2, division by zero or close to zero
    intersect_x = (B1C2 - C1B2) / (B1A2 - A1B2)
    intersect_y = (A1C2 - C1A2) / (A1B2 - B1A2)
    return intersect_x, intersect_y


def add_poles(coords1, coords2, pole1, pole2):
    coords1 = np.vstack([pole1, coords1[1:-1], pole2])
    coords2 = np.vstack([pole1, coords2[1:-1], pole2])
    result = np.stack(
        [coords1[:, 0], coords1[:, 1], coords2[:, 0], coords2[:, 1]], axis=1
    )
    return result, coords1, coords2


# MOMIA
def line_length(line):
    v1 = line[:-1]
    v2 = line[1:]
    d = v2 - v1
    return np.sum(np.sqrt(np.sum(d**2, axis=1)))


# MOMIA
def direct_intersect_distance(skeleton, contour):

    v1, v2 = contour[:-1], contour[1:]
    skel_x, skel_y = skeleton[1:-1].T
    intersect_x, intersect_y = intersect_matrix(skeleton[1:-1], contour)
    dx_v1 = intersect_x - v1.T[0][:, np.newaxis]
    dx_v2 = intersect_x - v2.T[0][:, np.newaxis]
    dy_v1 = intersect_y - v1.T[1][:, np.newaxis]
    dy_v2 = intersect_y - v2.T[1][:, np.newaxis]
    dx = dx_v1 * dx_v2
    dy = dy_v1 * dy_v2
    dist_x = skel_x[np.newaxis, :] - intersect_x
    dist_y = skel_y[np.newaxis, :] - intersect_y

    non_boundry_points = np.where(np.logical_and(dy > 0, dx > 0))
    dist_matrix = np.sqrt(dist_x**2 + dist_y**2)
    dist_matrix[non_boundry_points] = np.inf
    nearest_id_x = np.argsort(dist_matrix, axis=0)[:2]
    nearest_id_y = np.linspace(
        0, dist_matrix.shape[1] - 1, dist_matrix.shape[1]
    ).astype(int)
    dists = (
        dist_matrix[nearest_id_x[0], nearest_id_y]
        + dist_matrix[nearest_id_x[1], nearest_id_y]
    )
    return np.concatenate([[0], dists, [0]])


def contour2mesh(contour, cropped_mask):
    try:
        skeleton = extract_skeleton(cropped_mask)
    except Exception:
        bactoscoop_logger.debug("Skeletonization failed")
        skeleton = []
    if np.any(skeleton):
        try:
            extended_skeleton, pole1, pole2 = extend_skeleton(
                skeleton, contour, find_pole1=True, find_pole2=True
            )

            length = line_length(extended_skeleton)
            width = direct_intersect_distance(extended_skeleton, contour)
            if np.any(width):
                l1, l2, profile_mesh, midline = straighten_by_orthogonal_lines(
                    contour, extended_skeleton, length, width, unit_micron=0.5
                )
                result, l1, l2 = add_poles(l1, l2, pole1, pole2)
                return contour, result, midline
        except Exception:
            bactoscoop_logger.debug("width calculations for mesh creation failed")
            width = []
    return [], [], []


def draw_mask(contour, image):
    dilated_contour = expand_contour(contour, 0.5)
    mask_dilated = polygon2mask(image.shape, dilated_contour)
    return mask_dilated


def crop_image(image=None, contour=None, mask_to_crop=None, phase_img=None, pad=10):
    if contour is not None:
        cropped_contour = contour.copy()
        contour = contour.astype(int)
        x, y, w, h = crop(contour, pad, image)
        cropped_img = image[x : x + w, y : y + h]

        if phase_img is not None:
            cropped_phase_img = phase_img[x : x + w, y : y + h]
            # Perform registration between cropped_img and cropped_phase_img
            shift, error, _diff = phase_cross_correlation(
                cropped_phase_img, cropped_img, upsample_factor=20
            )
            if max(np.abs(shift)) <= 15:

                shift = np.array(shift)  # Ensure shift is in correct format
                cropped_img = shift_image(cropped_img, shift)
            else:
                cropped_img = image[x : x + w, y : y + h]
        cropped_mask = None
        cropped_contour -= [x, y]

    elif mask_to_crop is not None:
        x, y, w, h = crop(mask_to_crop, pad, mask_to_crop)
        cropped_mask = mask_to_crop[y : y + h, x : x + w]
        cropped_img = None
        cropped_contour = contour

    return cropped_img, cropped_mask, cropped_contour, x, y


def crop(to_be_cropped, pad, image):

    padding = pad  # Default padding value
    x, y, w, h = cv.boundingRect(
        to_be_cropped
    )  # Get the smallest bounding rectangle of the contour

    x = np.clip(x - padding, 0, image.shape[0])  # Adjust x-coordinate with padding
    y = np.clip(y - padding, 0, image.shape[1])  # Adjust y-coordinate with padding
    w = np.clip(w + 2 * padding, 0, image.shape[0] - x)  # Adjust width with padding
    h = np.clip(h + 2 * padding, 0, image.shape[1] - y)  # Adjust height with padding
    return x, y, w, h


def get_object_contours(mask, smoothing=0.1):  # x_offset, y_offset,
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    interpolated_contours = []

    for contour in contours:
        contour = np.array(contour)
        contourx = contour[:, 0, 1]
        contoury = contour[:, 0, 0]
        s = int(len(contourx) * smoothing)
        try:
            tck, u = interpolate.splprep([contourx, contoury], u=None, s=s, per=1)
            u_new = np.linspace(u.min(), u.max(), len(contourx))
            outx, outy = interpolate.splev(u_new, tck, der=0)
            interpolated_contour = np.array([outx, outy]).T
        except Exception as e:
            bactoscoop_logger.debug(f"{e}")
            continue
        interpolated_contours.append(interpolated_contour)

    return interpolated_contours


def get_cellular_mesh(masks, smoothing):
    meshdata = {}
    meshdataframe = pd.DataFrame()
    for cell in tqdm(range(1, np.max(masks))):
        temp_mask = np.where(masks == cell, 1, 0)
        temp_mask = cv.convertScaleAbs(temp_mask)

        _, cropped_mask, _, x, y = crop_image(mask_to_crop=temp_mask)

        cell_contours = get_object_contours(cropped_mask, smoothing=smoothing)
        for cn in cell_contours:
            contour, result, midline = contour2mesh(cn, cropped_mask)
            if np.any(contour):

                contour += [y, x]
                result += [y, x, y, x]
                midline += [y, x]

                meshdata[cell] = {
                    "contour": contour,
                    "mesh": result,
                    "midline": midline,
                }
        meshdataframe = pd.DataFrame.from_dict(
            meshdata, orient="index", columns=["contour", "mesh", "midline"]
        )
        meshdataframe.reset_index(drop=True, inplace=True)
    return meshdataframe


# MOMIA
def distance(v1, v2):
    # Euclidean distance of two points
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))


# MOMIA
def measure_width(extended_skeleton, smoothed_contour):
    length = line_length(extended_skeleton)
    d_perp = unit_perpendicular_vector(extended_skeleton, closed=False)
    width_list = []
    for i in range(1, len(extended_skeleton) - 1):
        xy = line_contour_intersection(
            extended_skeleton[i], d_perp[i] + extended_skeleton[i], smoothed_contour
        )
        coords = np.average(xy, axis=0)
        if (len(xy) == 2) and (np.isnan(coords).sum() == 0):
            width_list.append(distance(xy[0], xy[1]))
        else:
            raise ValueError(
                "Error encountered while computing line intersection points!"
            )
    return np.array([0] + width_list + [0]), length


def get_cell_pairs(pole_1, pole_2, cell_ID, maxdist=None):
    dm11 = distance_matrix(pole_1, pole_1)
    dm22 = distance_matrix(pole_2, pole_2)
    dm12 = distance_matrix(pole_1, pole_2)
    for i in cell_ID:
        dm11 = np.insert(
            dm11, i - 1, values=np.zeros((1, dm11.shape[1]), dtype=int), axis=0
        )
        dm11 = np.insert(dm11, i - 1, values=np.zeros(dm11.shape[0], dtype=int), axis=1)
        dm22 = np.insert(
            dm22, i - 1, values=np.zeros((1, dm22.shape[1]), dtype=int), axis=0
        )
        dm22 = np.insert(dm22, i - 1, values=np.zeros(dm22.shape[0], dtype=int), axis=1)
        dm12 = np.insert(
            dm12, i - 1, values=np.zeros((1, dm12.shape[1]), dtype=int), axis=0
        )
        dm12 = np.insert(dm12, i - 1, values=np.zeros(dm12.shape[0], dtype=int), axis=1)
    if maxdist != None:
        maxdistance = maxdist
        neighboring_cells_dm11 = np.argwhere(
            np.logical_and(dm11 > 0, dm11 <= maxdistance)
        )
        neighboring_cells_dm22 = np.argwhere(
            np.logical_and(dm22 > 0, dm22 <= maxdistance)
        )
        neighboring_cells_dm12 = np.argwhere(
            np.logical_and(dm12 > 0, dm12 <= maxdistance)
        )
        nc_concat = np.concatenate(
            (neighboring_cells_dm11, neighboring_cells_dm22, neighboring_cells_dm12),
            axis=0,
        )
        unique_nc1 = np.unique(nc_concat, axis=0)
        unique_nc2 = unique_nc1[unique_nc1[:, 0] != unique_nc1[:, 1]]
    else:
        # get index list of cell couples. Loop decides the distance for when more than 2 cells from couples with other cells
        maxdistance = 1
        while True:  # or maxdist = 8
            neighboring_cells_dm11 = np.argwhere(
                np.logical_and(dm11 > 0, dm11 <= maxdistance)
            )
            neighboring_cells_dm22 = np.argwhere(
                np.logical_and(dm22 > 0, dm22 <= maxdistance)
            )
            neighboring_cells_dm12 = np.argwhere(
                np.logical_and(dm12 > 0, dm12 <= maxdistance)
            )
            nc_concat = np.concatenate(
                (
                    neighboring_cells_dm11,
                    neighboring_cells_dm22,
                    neighboring_cells_dm12,
                ),
                axis=0,
            )
            unique_nc1 = np.unique(nc_concat, axis=0)
            unique_nc2 = unique_nc1[unique_nc1[:, 0] != unique_nc1[:, 1]]
            unique_values, inverse_indices = np.unique(unique_nc2, return_inverse=True)
            counts = np.bincount(inverse_indices)
            # Check if the current neighboring cells are already in the set of unique cells
            if counts == []:
                print("No neighboring cells present in the image")
                break
            try:
                if np.max(counts) > 2:
                    print(
                        f"Found triple neighboring cells after {maxdistance} distance units"
                    )
                    maxdistance = maxdistance
                    break
            except ValueError:
                pass

            # Increment the maximum distance to search for
            maxdistance += 1
            if maxdistance > 10:
                print("pole distance to reach 4 stitched cells exceeds 20")
                break
            if counts == []:
                unique_nc2 = []
            else:
                neighboring_cells_dm11 = np.argwhere(
                    np.logical_and(dm11 > 0, dm11 <= maxdistance)
                )
                neighboring_cells_dm22 = np.argwhere(
                    np.logical_and(dm22 > 0, dm22 <= maxdistance)
                )
                neighboring_cells_dm12 = np.argwhere(
                    np.logical_and(dm12 > 0, dm12 <= maxdistance)
                )
                nc_concat = np.concatenate(
                    (
                        neighboring_cells_dm11,
                        neighboring_cells_dm22,
                        neighboring_cells_dm12,
                    ),
                    axis=0,
                )
                unique_nc1 = np.unique(nc_concat, axis=0)
                unique_nc2 = unique_nc1[unique_nc1[:, 0] != unique_nc1[:, 1]]
    return unique_nc2


def mesh2contour(x1, y1, x2, y2):
    x2f = np.flip(x2)
    y2f = np.flip(y2)
    # Concatenate the x and y coordinates
    xspline = np.concatenate((x2f[1:], x1[1:]))
    yspline = np.concatenate((y2f[1:], y1[1:]))

    tck, u = interpolate.splprep(np.array([xspline, yspline]), k=3, s=2, per=1)
    u_new = np.linspace(u.min(), u.max(), 200)
    outx, outy = interpolate.splev(u_new, tck)

    return np.array([outx, outy]).T


def split_point(x1, y1, x2, y2, ctpos):
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    # ctpos is the index at which the cell is most constricted and where the algorith will split the cell
    xc1 = x1[ctpos]
    yc1 = y1[ctpos]
    xc2 = x2[ctpos]
    yc2 = y2[ctpos]
    newpole_x = (xc1 + xc2) / 2
    newpole_y = (yc1 + yc2) / 2

    xt1, yt1 = x1[ctpos + 1], y1[ctpos + 1]
    distt = dist[ctpos + 1]
    # Use indexing to get the x and y coordinates separately
    xt13, yt13 = (
        np.array([xt1, yt1])
        + (distt / 3) * np.array([dx[ctpos + 1], dy[ctpos + 1]]) / distt
    )
    xt23, yt23 = (
        np.array([xt1, yt1])
        + (distt * (2 / 3)) * np.array([dx[ctpos + 1], dy[ctpos + 1]]) / distt
    )

    xs1, ys1 = x1[ctpos - 1], y1[ctpos - 1]
    dists = dist[ctpos - 1]

    # Use indexing to get the x and y coordinates separately
    xs13, ys13 = (
        np.array([xs1, ys1])
        + (dists / 3) * np.array([dx[ctpos - 1], dy[ctpos - 1]]) / dists
    )
    xs23, ys23 = (
        np.array([xs1, ys1])
        + (dists * (2 / 3)) * np.array([dx[ctpos - 1], dy[ctpos - 1]]) / dists
    )

    return (
        newpole_x,
        newpole_y,
        np.array([[xt13, yt13, xt23, yt23]]),
        np.array([[xs13, ys13, xs23, ys23]]),
    )


# calculates the constriction degree in relative and absolute value,
# and the relative position along the length of the cell
# absolute constr degree needs to be multiplied by the width
def constr_degree_single_cell_min(
    intensity, new_step_length, upper_limit=0.75, lower_limit=0.25
):
    minsize = 0
    minsizeabs = 0
    ctpos = []
    ctposr = np.nan

    # Determine indices for upper and lower limits
    start_index = int(lower_limit * len(intensity))
    end_index = int(upper_limit * len(intensity))
    # Slice the intensity array to focus on the specified section
    intensity_section = intensity[start_index:end_index]

    try:
        minima = np.concatenate(
            (
                [False],
                (intensity_section.T[1:-1] < intensity_section.T[:-2])
                & (intensity_section.T[1:-1] < intensity_section.T[2:]),
                [False],
            )
        )
    except AttributeError:
        return minsize, np.nan, minsizeabs, ctpos

    if all(not x for x in minima) or np.sum(intensity_section.T) == 0:
        return minsize, np.nan, minsizeabs, ctpos

    index, dh, dhi, hgt = identify_minima(intensity_section, minima)
    fix = np.argmax(dh)

    minsizeabs = dhi[fix]
    hgt_fix = hgt[fix]

    if hgt_fix != 0:
        minsize = minsizeabs / hgt_fix

    ctpos = index[fix]
    ctposr = (start_index + ctpos) / len(intensity)

    return minsize, ctposr, minsizeabs, ctpos


def interp2d(image):
    ff = interpolate.RectBivariateSpline(
        range(image.shape[0]), range(image.shape[1]), image, kx=1, ky=1
    )
    return ff


def get_image_for_frame(df_nr, images):
    return images[df_nr]


def get_cell_mask(contour, midline, image):
    cropped_img, cropped_mask, cropped_contour, x, y = crop_image(
        image=image, contour=contour, mask_to_crop=None
    )
    cropped_midline = midline.copy()
    cropped_midline -= [x, y]
    mask = draw_mask(cropped_contour, cropped_img)
    return mask, cropped_img, cropped_contour, cropped_midline, x, y


# %% Object detection and mesh creation functions
def keep_significant_masks(
    nucleoid_mask, cropped_mask, min_overlap_ratio=0.01, max_external_ratio=0.1
):
    # Find connected components (blobs) in the nucleoid_mask
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        nucleoid_mask.astype(np.uint8)
    )

    # Create a mask to keep blobs with significant overlap
    significant_mask = np.zeros_like(nucleoid_mask, dtype=np.uint8)

    # Get the total area of the cropped_mask
    total_cropped_mask_area = np.sum(cropped_mask)

    for label in range(1, num_labels):  # Skip label 0 as it represents the background
        # Extract the blob region for the current label
        blob_mask = (labels == label).astype(np.uint8)

        # Calculate the overlap with the cropped_mask
        overlap_area = np.sum(np.logical_and(blob_mask, cropped_mask))

        # Calculate the overlap ratio
        overlap_ratio = overlap_area / total_cropped_mask_area

        # Calculate the external ratio (portion outside of cropped_mask)
        external_area = np.sum(np.logical_and(blob_mask, ~cropped_mask))
        external_ratio = external_area / np.sum(blob_mask)
        # Keep blobs that have a significant overlap with the cropped_mask
        # and don't exceed the maximum external ratio
        if overlap_ratio >= min_overlap_ratio and external_ratio <= max_external_ratio:
            significant_mask |= blob_mask

    return significant_mask


def get_object_masks(contour, object_contours, object_midlines, object_meshes, signal):

    masks = []
    cropped_img, cropped_mask, cropped_contour, x, y = crop_image(
        image=signal, contour=contour, mask_to_crop=None
    )
    cropped_object_contours, cropped_object_midlines, cropped_object_meshes = (
        get_cropped_object_data(object_contours, object_midlines, object_meshes, x, y)
    )
    for cropped_object_contour in cropped_object_contours:

        temp_masks = draw_mask(cropped_object_contour, cropped_img)
        masks.append(temp_masks)

    return (
        masks,
        cropped_img,
        cropped_object_contours,
        cropped_object_midlines,
        cropped_object_meshes,
        x,
        y,
    )


def get_subcellular_objects(
    contour,
    signal,
    image,
    cell_id,
    smoothing,
    log_sigma,
    kernel_width,
    min_overlap_ratio=0.01,
    max_external_ratio=0.1,
):

    cropped_img, _, cropped_contour, x_offset, y_offset = crop_image(
        image=signal, contour=contour, mask_to_crop=None, phase_img=image
    )
    
    
    cropped_mask = draw_mask(cropped_contour, cropped_img)
    bgr = median_background(cropped_img, cropped_mask)
    cropped_img_bgr = np.maximum(cropped_img - bgr, 0)
    
    expanded_contour = expand_contour(cropped_contour, 5)
    cropped_img_bgr_zero = set_bg_zero(expanded_contour, cropped_img_bgr)
    # Apply Laplacian of Gaussian (LoG) filter to the cropped_img
    log_filtered_img = gaussian_laplace(cropped_img_bgr_zero, sigma=log_sigma)

    threshold_value = threshold_otsu(log_filtered_img)

    object_mask = log_filtered_img < threshold_value

    kernel = np.ones(
        (kernel_width, kernel_width), np.uint8
    )  # You can adjust the kernel size as needed
    object_mask = cv.dilate(object_mask.astype(np.uint8), kernel, iterations=1)

    object_mask = keep_significant_masks(
        object_mask, cropped_mask, min_overlap_ratio, max_external_ratio
    )

    if np.any(object_mask == 1):
        object_contours = get_object_contours(
            object_mask, smoothing
        )  # , x_offset, y_offset
    else:
        object_contours = []

    return object_contours, cropped_img, object_mask, x_offset, y_offset


def get_object_mesh(
    contour,
    signal,
    image,
    cell_id,
    smoothing,
    log_sigma,
    kernel_width,
    min_overlap_ratio,
    max_external_ratio,
):
    meshdata = {
        "cell_id": cell_id,
        "object_contour": [],
        "cropped_object_contour": [],
        "object_mesh": [],
        "object_midline": [],
    }
    object_contours, cropped_img, object_mask, x, y = get_subcellular_objects(
        contour,
        signal,
        image,
        cell_id,
        smoothing,
        log_sigma,
        kernel_width,
        min_overlap_ratio,
        max_external_ratio,
    )

    if object_contours:

        for cn in object_contours:
            meshdata["object_contour"].append(cn + [x, y])
            meshdata["cropped_object_contour"].append(cn)
            temp_mask = draw_mask(cn, object_mask)
            contour, result, midline = contour2mesh(cn, temp_mask)
            # Store the object_contours already

            if np.any(result):

                result += [x, y, x, y]
                midline += [x, y]

                # Store the rest of the data in case of succesful execution
                meshdata["object_mesh"].append(result)
                meshdata["object_midline"].append(midline)

            else:

                meshdata["object_mesh"].append(np.array(result))
                meshdata["object_midline"].append(np.array(midline))

    return meshdata


# %% File readers/handlers
from natsort import natsorted
def read_channels(folder_path, images_dict, channel_list=None):
    if channel_list is None:
        # If no suffixes are provided, assume an empty list
        channel_list = []

    for channel in channel_list:
        matching_files = natsorted([
            f
            for f in os.listdir(folder_path)
            if f.endswith(f"{channel}.tif") or f.endswith(f"{channel}.tiff")
        ])

        if matching_files:
            images_dict[channel] = [
                tifffile.imread(os.path.join(folder_path, tiff_file))
                for tiff_file in matching_files
            ]
            bactoscoop_logger.info(
                f"Images with specified suffixes {channel} found in folder: {folder_path}"
            )

    if not any(images_dict.values()):
        raise ValueError(
            f"\nNo images with specified suffixes ({channel_list}) found in folder: {folder_path}"
        )

    return images_dict


def read_tiff_folder(folder_path, suffix="", include_paths=False):
    # Get a list of all TIFF files in the folder
    tiff_files = natsorted([
        f
        for f in os.listdir(folder_path)
        if f.endswith(f"{suffix}.tif") or f.endswith(f"{suffix}.tiff")
    ])
    if not tiff_files:
        # Raise an exception if no TIFF files are found in the folder
        raise ValueError("\nNo TIFF files found in folder: " + folder_path)

    images = []
    file_names = []
    paths = [] if include_paths else None

    for tiff_file in tiff_files:
        tiff_path = os.path.join(folder_path, tiff_file)
        image = tifffile.imread(tiff_path)
        images.append(image)
        file_names.append(tiff_file)
        if include_paths:
            paths.append(tiff_path)

    if len(images) == 1:
        bactoscoop_logger.info(f"Only one TIFF file found in folder: {folder_path}")
        images = np.expand_dims(images[0], axis=0)
        file_names = [file_names[0]]
        if include_paths:
            paths = [paths[0]]

    else:
        bactoscoop_logger.info(f"{len(images)} TIFF files found in folder: {folder_path}")

    if include_paths:
        return images, file_names, paths
    else:
        return images, file_names


# %% Contour and mesh operations
def separate_meshdata(df_nr, cell_nr, data):
    x1 = data[df_nr]["mesh"][cell_nr][:, 0]
    y1 = data[df_nr]["mesh"][cell_nr][:, 1]
    x2 = data[df_nr]["mesh"][cell_nr][:, 2]
    y2 = data[df_nr]["mesh"][cell_nr][:, 3]
    contour = data[df_nr]["contour"][cell_nr]
    return x1, y1, x2, y2, contour


def separate_singleframe_meshdata(cell_id, data):
    row = data[data.index == cell_id].iloc[0]
    x1 = row["mesh"][:, 0]
    y1 = row["mesh"][:, 1]
    x2 = row["mesh"][:, 2]
    y2 = row["mesh"][:, 3]
    contour = row["contour"]
    midline = row["midline"]
    return x1, y1, x2, y2, contour, midline


def get_cropped_object_data(cnts, mdls, mshs, x, y):
    cropped_cnts = []
    cropped_mdls = []
    cropped_mshs = []

    for cn, ml, msh in zip(cnts, mdls, mshs):
        crop_cn = cn.copy()
        crop_ml = ml.copy()
        crop_msh = msh.copy()

        crop_cn -= [x, y]
        crop_ml -= [x, y]
        crop_msh -= [x, y, x, y]

        cropped_cnts.append(crop_cn)
        cropped_mdls.append(crop_ml)
        cropped_mshs.append(crop_msh)

    return cropped_cnts, cropped_mdls, cropped_mshs


def get_cropped_cell_data(cnt, mdl, msh, x, y):
    cropped_cn = cnt.copy()
    cropped_ml = mdl.copy()
    cropped_msh = msh.copy()

    cropped_cn -= [x, y]
    cropped_ml -= [x, y]
    cropped_msh -= [x, y, x, y]

    return cropped_cn, cropped_ml, cropped_msh


def get_uncropped_cell_data(cnt, mdl, msh, x, y):
    uncropped_cn = cnt.copy()
    uncropped_ml = mdl.copy()
    uncropped_msh = msh.copy()

    uncropped_cn += [x, y]
    uncropped_ml += [x, y]
    uncropped_msh += [x, y, x, y]

    return uncropped_cn, uncropped_ml, uncropped_msh


def mesh2midline(x1, y1, x2, y2):
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    line = np.array([x, y]).T
    line = spline_approximation(line, n=len(x1), smooth_factor=3, closed=False)
    return line


def get_weighted_intprofile(intensity, width):

    if intensity.shape[0] == width.shape[0]:
        return intensity * width

    else:
        return []


def get_width(x1, y1, x2, y2):
    width_not_ordered = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return width_not_ordered


def split_mesh2mesh(x1, y1, x2, y2):

    contour = mesh2contour(x1, y1, x2, y2)

    try:
        width, widthno = get_avg_width_no_px(x1, y1, x2, y2)

        length = np.sum(get_step_length_no_px(x1, y1, x2, y2))

        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        line = np.array([x, y]).T
        midline, pole1, pole2 = extend_skeleton(line[4:-4], contour)
        l1, l2, profile_mesh, midline = straighten_by_orthogonal_lines(
            contour, midline, length, width, unit_micron=0.5
        )
        result, l1, l2 = add_poles(l1, l2, pole1, pole2)
    except Exception as e:
        bactoscoop_logger.debug(f"Exception occurred: {e}")
        pass  # needs logging
    return result, contour, midline


# MOMIA
def get_profile_mesh(mesh, width, micron_unit=1):
    l1 = mesh[:, :2]
    l2 = mesh[:, 2:]
    sorted_width = sorted(width, reverse=True)
    width = sum(sorted_width[: math.floor(len(sorted_width) / 3)]) / math.floor(
        len(sorted_width) / 3
    )
    N_width = int(round(np.median(width) / 0.05))
    dl = (l2 - l1) / N_width
    mult_mat = np.tile(np.arange(N_width + 1), (len(l1), 1))
    mat_x = l1[:, 0][:, np.newaxis] + mult_mat * dl[:, 0][:, np.newaxis]
    mat_y = l1[:, 1][:, np.newaxis] + mult_mat * dl[:, 1][:, np.newaxis]
    profile_mesh = np.array([mat_x, mat_y])
    return profile_mesh


# MOMIA
def expand_contour(contour, scale=1):
    """
    enlarges contour
    :param contour:
    :param scale:
    :return:
    """
    area = 0.5 * np.sum(np.diff(contour[:, 0]) * (contour[:-1, 1] + contour[1:, 1]))
    if area < 0:
        # If the area is negative, flip the sign of the unit perpendicular vector
        dxy = unit_perpendicular_vector(contour, closed=True)
    else:
        # Otherwise, use the unit perpendicular vector as is
        dxy = unit_perpendicular_vector(contour, closed=True) * (-1)
    # dxy = unit_perpendicular_vector(contour, closed=True)

    return contour - (scale * dxy)


def erode_contour(contour, scale=1):
    """
    shrinks contour
    :param contour:
    :param scale:
    :return:
    """
    area = 0.5 * np.sum(np.diff(contour[:, 0]) * (contour[:-1, 1] + contour[1:, 1]))
    if area < 0:
        # If the area is negative, flip the sign of the unit perpendicular vector
        dxy = unit_perpendicular_vector(contour, closed=True)
    else:
        # Otherwise, use the unit perpendicular vector as is
        dxy = unit_perpendicular_vector(contour, closed=True) * (-1)
    # dxy = unit_perpendicular_vector(contour, closed=True)

    return contour + (scale * dxy)


### ------------------------------------------------------------------------------------------------
### Background subtraction
# calculates the average pixel value of the background region in an image,
# given a binary cell mask.
def mean_background(image, cell_mask):
    # Create a mask for the background region
    background_mask = np.logical_not(cell_mask)

    # Calculate the average pixel value of the background
    # =>indexing the image array with the background_mask, which selects
    #  only the pixel values
    bgr = np.mean(image[background_mask])
    return bgr


def median_background(image, cell_mask):
    # Create a mask for the background region
    background_mask = np.logical_not(cell_mask)

    # Calculate the average pixel value of the background
    # =>indexing the image array with the background_mask, which selects
    #  only the pixel values
    bgr = np.median(image[background_mask])
    return bgr


def poly_func(coords, a, b, c, d, e, f):
    x, y = coords
    return a + b * x + c * y + d * x**2 + e * x * y + f * y**2


def construct_bg_function(image, seg_mat, sampling_interval=None, plotF=False):
    """
    Reconstructs the background of an image with a second-degree polynomial.

    Parameters:
        image (ndarray): An n by m matrix representing an image.
        seg_mat (ndarray): An n by m matrix where non-zero-values represent the pixel mask of objects.
        sampling_interval (int, optional): Interval between two samples taken for the approximation.
                                            Default is calculated as size(image, 1) / 64.
        plotF (bool, optional): If True, plot the samples and the reconstruction. Default is False.

    Returns:
        bg_function (function): A function F(x Col,y Row) which calculates the background value for the desired point.
        bg_matrix (ndarray): The matrix as the background noise to be subtracted.
    """
    if sampling_interval is None:
        sampling_interval = max(1, image.shape[0] // 64)

    bg_samples = np.zeros_like(seg_mat, dtype=bool)
    bg_samples[::sampling_interval, ::sampling_interval] = ~seg_mat[
        ::sampling_interval, ::sampling_interval
    ].astype(bool)

    bg_samples_for_approx = np.where(bg_samples, image, 0)

    y, x = np.nonzero(bg_samples)
    z = bg_samples_for_approx[y, x]

    popt, _ = curve_fit(poly_func, (x, y), z, p0=(0, 0, 0, 0, 0, 0))

    bg_function = lambda c, r: poly_func((c, r), *popt)

    bg_matrix = np.fromfunction(np.vectorize(bg_function), image.shape, dtype=float)

    if plotF:
        xx, yy = np.meshgrid(
            np.arange(0, bg_samples.shape[1], 10), np.arange(0, bg_samples.shape[0], 10)
        )
        zz = bg_function(xx, yy)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, zz, alpha=0.5)
        ax.scatter(x, y, z, color="r")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.show()

    return bg_function, bg_matrix


###  -------------------------------------------------------------------------------------------------
# %% Morphological features
def get_step_length(x1, y1, x2, y2, px):
    dx = x1[1:] + x2[1:] - x1[:-1] - x2[:-1]
    dy = y1[1:] + y2[1:] - y1[:-1] - y2[:-1]
    return (np.sqrt(dx**2 + dy**2) / 2) * px


def get_step_length_no_px(x1, y1, x2, y2):
    dx = x1[1:] + x2[1:] - x1[:-1] - x2[:-1]
    dy = y1[1:] + y2[1:] - y1[:-1] - y2[:-1]
    return np.sqrt(dx**2 + dy**2) / 2


def get_length(step_length):
    return np.sum(step_length)


def get_avg_width(x1, y1, x2, y2, px):
    width_not_ordered = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    sorted_width = sorted(width_not_ordered, reverse=True)
    width = sum(sorted_width[: math.floor(len(sorted_width) / 3)]) / math.floor(
        len(sorted_width) / 3
    )
    return width * px, width_not_ordered * px


def get_avg_width_no_px(x1, y1, x2, y2):
    width_not_ordered = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    sorted_width = sorted(width_not_ordered, reverse=True)
    width = sum(sorted_width[: math.floor(len(sorted_width) / 3)]) / math.floor(
        len(sorted_width) / 3
    )
    return width, width_not_ordered


def get_area(contour, px):
    poly = Polygon(contour)
    area = poly.area
    return area * px * px


def get_volume(x1, y1, x2, y2, step_length, px):
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    volume = np.trapz((np.pi * (d / 2) ** 2), dx=step_length)
    return volume * px * px


def get_surface_area(width_no, step_length):
    widths = width_no[1:]
    surface_areas = 2 * np.pi * (widths / 2) * step_length
    total_surface_area = np.sum(surface_areas)
    return total_surface_area


def get_surface_area_over_volume(sa, vol):
    return sa / vol


def get_cell_perimeter_measurements(contour, area, px):
    v1 = contour[:-1]
    v2 = contour[1:]
    d = v2 - v1
    perimeter = np.sum(np.sqrt(np.sum(d**2, axis=1))) * px
    circularity = (4 * np.pi * area) / (perimeter) ** 2
    compactness = (perimeter**2) / area
    sphericity = (np.pi * 1.5 * (perimeter / (2 * np.pi)) ** 1.5) / area
    return perimeter, circularity, compactness, sphericity


# cell width variability calculated based on the 50% highest cell widths
def get_cell_width_variability(width_no):
    sorted_width = sorted(width_no, reverse=True)
    half_idx = len(sorted_width) // 2
    half_width = sorted_width[:half_idx]
    width_var = np.std(half_width) / np.mean(half_width)
    return width_var


def get_curvature_characteristics(contour, px):
    dx = np.gradient(contour[:, 0] * px)
    dy = np.gradient(contour[:, 1] * px)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2) ** 1.5
    max_c = np.round(np.max(curvature), 8)
    min_c = np.round(np.min(curvature), 8)
    mean_c = np.round(np.nanmean(curvature), 8)
    std_c = np.round(np.std(curvature), 8)
    return curvature, max_c, min_c, mean_c, std_c


def get_total_phaco_intensity(contour, shape, interp2d):

    mask = polygon2mask(shape, contour)
    coords = np.column_stack(np.where(mask)).astype(int)
    values = interp2d.ev(coords[:, 0], coords[:, 1])
    return np.sum(values), np.max(values), np.mean(values)
    # Calculate the total phase intensity within the contour mask


def get_contour_intensity(contour, interp2d):
    data = interp2d.ev(contour.T[0], contour.T[1])
    return data


def find_signal_peaks(signal, maximum):
    peaks, _ = find_peaks(signal, prominence=0.5, height=(maximum * 0.5))
    return len(peaks)


def measure_contour_variability(signal):
    extra_contour_intensities = np.concatenate([signal, signal[:10]])
    contour_intensities_variability = np.array(
        [
            np.std(extra_contour_intensities[ss : ss + 10])
            for ss in range(1, len(extra_contour_intensities) - 10)
        ]
    )
    return contour_intensities_variability


def get_kurtosis(signal):
    return kurtosis(signal)


def get_skew(signal):
    return skew(signal)


def sinuosity(midline):
    if (midline[0] - midline[-1]).sum() == 0:
        raise ValueError("Unable to calculate sinuosity for a closed contour.")

    end_to_end_dist = distance(midline[0], midline[-1])
    length = line_length(midline)
    sinuosity = round(length / end_to_end_dist, 3)

    if sinuosity < 1:
        sinuosity = 1.0

    return sinuosity


def get_area_asymmetry(step_length, width_no):
    result = [step * width for step, width in zip(step_length, width_no)]

    # Calculate the two areas
    area1 = sum(result[: len(result) // 2])
    area2 = sum(result[len(result) // 2 :])

    if area1 == area2:
        return 1.0
    # Determine which area is smaller and which is larger
    smaller_area = min(area1, area2)
    larger_area = max(area1, area2)

    # Calculate the ratio of the smaller area over the larger area
    asymmetry_ratio = smaller_area / larger_area

    return asymmetry_ratio


#  ------------------------------------------------------------------------------------------------
# %% Object features (multiple contours)
def get_object_params(object_contours, px):
    length_list, width_list, box_list, x_y_list = [], [], [], []

    for object_contour in object_contours:
        object_contour = object_contour.astype(np.float32)
        rect = cv.minAreaRect(object_contour)
        box = cv.boxPoints(rect).tolist()
        x_y, (width, length), _ = rect
        x_y_list.append(x_y)
        box_list.append(box)
        length_list.append(max(width, length) * px)
        width_list.append(min(width, length) * px)

    total_length = sum(length_list)
    average_width = sum(width_list) / len(width_list)

    return total_length, average_width


def get_object_width_variability(widths_no):

    width_var_list = []

    for width_no in widths_no:
        width_var = get_cell_width_variability(width_no)
        width_var_list.append(width_var)

    avg_width_var = [np.nan] if not width_var_list else np.mean(width_var_list)

    return width_var_list, avg_width_var


def get_mesh_coordinates(mesh):
    x1 = mesh[:, 0]
    y1 = mesh[:, 1]
    x2 = mesh[:, 2]
    y2 = mesh[:, 3]
    return x1, y1, x2, y2


def get_object_rect_length(nucleoid_contours, px):
    rect_length_list = []
    rect_width_list = []

    for nucleoid_contour in nucleoid_contours:
        nucleoid_contour = nucleoid_contour.astype(np.float32)

        x_y, width_height, angle_of_rotation = cv.minAreaRect(nucleoid_contour)

        length = max(width_height) * px
        rect_length_list.append(length)
        width = min(width_height) * px
        rect_width_list.append(width)

    total_rect_length = sum(rect_length_list)
    # Calculate average width, handling division by zero
    average_rect_width = [np.nan] if not rect_width_list else np.mean(rect_width_list)

    return rect_length_list, rect_width_list, total_rect_length, average_rect_width


def get_additional_regionprops_features(mask, px):
    mask = mask.astype(bool)
    props = regionprops(mask.astype(int))[0]
    # Calculate Feret diameters
    coords = np.column_stack(np.where(mask))
    distances1 = spatial.distance.pdist(coords)
    max_feret_diameter = np.max(distances1) * px

    # Calculate radii
    center = props.centroid
    y, x = np.nonzero(mask)
    distances2 = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2) * px
    max_radius = np.max(distances2)
    mean_radius = np.mean(distances2)
    median_radius = np.median(distances2)

    # Create features dictionary
    features = {
        "F_MAX_FERET_DIAMETER": max_feret_diameter,
        "F_EQUIVALENT_DIAMETER": props.equivalent_diameter * px,
        "F_MAXIMUM_RADIUS": max_radius,
        "F_MEAN_RADIUS": mean_radius,
        "F_MEDIAN_RADIUS": median_radius,
        "F_EXTENT": props.extent,
    }

    # Add moments and other properties
    moment_properties = [
        "moments",
        "moments_central",
        "moments_normalized",
        "moments_hu",
        "inertia_tensor",
        "inertia_tensor_eigvals",
    ]

    for prop in moment_properties:
        if prop in ["moments_hu", "inertia_tensor_eigvals"]:
            for i, value in enumerate(getattr(props, prop)):
                features[f"{prop}_{i}"] = value
        else:
            moments = getattr(props, prop)
            for i in range(moments.shape[0]):
                for j in range(moments.shape[1]):
                    # Skip moments_normalized for indices (0,1), (0,0), and (1,0)
                    if prop == "moments_normalized" and (
                        i == 0 and (j == 1 or j == 0) or i == 1 and j == 0
                    ):
                        continue
                    features[f"{prop}_{i}_{j}"] = moments[i, j]

    return features


def get_object_avg_width(nucleoid_masks, px):

    nucleoid_avg_width_list = []
    width_not_ordered_list = []

    for nucleoid_mask in nucleoid_masks:

        # Convert nucleoid_mask to 8-bit unsigned integer type
        nucleoid_mask = nucleoid_mask.astype(np.uint8)
        # Ensure both arrays have the same data type
        distance = cv.distanceTransform(
            nucleoid_mask, distanceType=cv.DIST_L2, maskSize=5
        ).astype(np.float64)
        # Extract values at skeleton locations
        skeleton = skeletonize(nucleoid_mask)

        width = distance[skeleton] * 2 * px
        width_not_ordered_list.append(width)
        nucleoid_avg_width_list.append(np.mean(width))

    return np.mean(nucleoid_avg_width_list), width_not_ordered_list


def get_object_aspect_ratio(rect_length_lists, rect_width_lists):

    aspect_ratio_list = []

    for length_list, width_list in zip(rect_length_lists, rect_width_lists):
        aspect_ratio = width_list / length_list
        aspect_ratio_list.append(aspect_ratio)
    avg_aspect_ratio = [np.nan] if not aspect_ratio_list else np.mean(aspect_ratio_list)

    return aspect_ratio_list, avg_aspect_ratio


def get_object_area(contours, px):

    area_list = []
    for contour in contours:

        poly = Polygon(contour)
        area = poly.area
        area_list.append(area * px * px)

    total_area = np.sum(area_list)
    return area_list, total_area


def get_object_curvature_characteristics(contours, px):
    curvatures = []
    max_c_list = []
    min_c_list = []
    mean_c_list = []
    std_c_list = []

    for contour in contours:
        # Calculate gradients of x and y coordinates
        curvature, max_c, min_c, mean_c, std_c = get_curvature_characteristics(
            contour, px
        )
        curvatures.append(curvature)
        max_c_list.append(max_c)
        min_c_list.append(min_c)
        mean_c_list.append(mean_c)
        std_c_list.append(std_c)

    avg_mean_c = [np.nan] if not mean_c_list else np.mean(mean_c_list)
    avg_std_c = [np.nan] if not std_c_list else np.mean(std_c_list)

    return (
        curvatures,
        max_c_list,
        min_c_list,
        mean_c_list,
        std_c_list,
        avg_mean_c,
        avg_std_c,
    )


def get_object_perimeter_measurements(contours, areas, px):
    perimeter_list = []
    circularity_list = []
    compactness_list = []
    sphericity_list = []

    for contour, area in zip(contours, areas):
        perimeter, circularity, compactness, sphericity = (
            get_cell_perimeter_measurements(contour, area, px)
        )

        perimeter_list.append(perimeter)
        circularity_list.append(circularity)
        compactness_list.append(compactness)
        sphericity_list.append(sphericity)

    avg_perimeter = [np.nan] if not perimeter_list else np.mean(perimeter_list)
    avg_circularity = [np.nan] if not circularity_list else np.mean(circularity_list)
    avg_compactness = [np.nan] if not compactness_list else np.mean(compactness_list)
    avg_sphericity = [np.nan] if not sphericity_list else np.mean(sphericity_list)

    return (
        perimeter_list,
        circularity_list,
        compactness_list,
        sphericity_list,
        avg_perimeter,
        avg_circularity,
        avg_compactness,
        avg_sphericity,
    )


def get_cell_convexity(mask):
    labels = label(mask)

    properties = regionprops(labels)
    prop = properties[0]

    image_perimeter = prop.perimeter
    image_eccentricity = prop.eccentricity
    image_solidity = prop.solidity
    convex_image_perimeter = regionprops(prop.convex_image * 1)[0].perimeter
    convexity = round(convex_image_perimeter / image_perimeter, 3)

    return image_eccentricity, image_solidity, convexity


def get_object_convexity(cropped_masks):

    convexity_list = []
    eccentricity_list = []
    solidity_list = []

    for nucleoid_mask in cropped_masks:

        ecc, sol, conv = get_cell_convexity(nucleoid_mask)

        convexity_list.append(conv)
        eccentricity_list.append(ecc)
        solidity_list.append(sol)

        avg_convexity = [np.nan] if not convexity_list else np.mean(convexity_list)
        avg_eccentricity = (
            [np.nan] if not eccentricity_list else np.mean(eccentricity_list)
        )
        avg_solidity = [np.nan] if not solidity_list else np.mean(solidity_list)

    return (
        convexity_list,
        eccentricity_list,
        solidity_list,
        avg_convexity,
        avg_eccentricity,
        avg_solidity,
    )


def get_approx_nucleoid_volume(cell_volume, nucleoid_areas, cell_area):

    nucleoid_volume_list = []

    for nucleoid_area in nucleoid_areas:
        # The power 3/2 was used to convert the estimated nucleoid area fraction into a volume fraction.
        nucleoid_volume = cell_volume * (nucleoid_area / cell_area) ** (3 / 2)
        # https://doi.org/10.1016/j.cell.2021.05.037
        nucleoid_volume_list.append(nucleoid_volume)
        total_volume = np.sum(nucleoid_volume_list)

    return nucleoid_volume_list, total_volume


def get_nucleoid_width_variability(widths):
    # Because the CV is unitless and usually expressed as a percentage, it is used instead of the SD to compare the spread of data
    # sets that have different units of measurements or have the same units of measurements but differs greatly in magnitude.
    # Ratio between the standard deviation and the mean ( which in some case could be negative)

    width_var_list = []
    if widths:

        for width in widths:
            width_var = np.std(width) / np.mean(width)
            width_var_list.append(width_var)

    avg_width_var = [np.nan] if not width_var_list else np.mean(width_var_list)

    return width_var_list, avg_width_var


def get_obj_mesh_coords(nucleoid_meshes):
    x1_list, y1_list, x2_list, y2_list = [], [], [], []

    for nucleoid_mesh in nucleoid_meshes:
        if np.any(nucleoid_mesh):
            x1 = nucleoid_mesh[:, 0]
            y1 = nucleoid_mesh[:, 1]
            x2 = nucleoid_mesh[:, 2]
            y2 = nucleoid_mesh[:, 3]

            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
        else:
            x1_list.append([])
            y1_list.append([])
            x2_list.append([])
            y2_list.append([])

    return x1_list, y1_list, x2_list, y2_list


def measure_along_objects_midline_interp2d(
    midlines, im_interp2d, width=7, subpixel=0.5
):

    prf_list = []

    for midline in midlines:
        if np.any(midline):
            # Calculate a unit vector perpendicular to the midline
            unit_dxy = unit_perpendicular_vector(midline, closed=False)

            # Calculate a normalized vector for the width and subpixel resolution
            width_normalized_dxy = unit_dxy * subpixel
            # It prepares the vectors for use in generating profiles at varying distances from the midline.

            # Interpolate image values along the midline
            data = im_interp2d.ev(midline.T[0], midline.T[1])

            # Iterate over width steps and calculate profile values
            for i in range(1, 1 + int(width * 0.5 / subpixel)):

                dxy = width_normalized_dxy * i
                # This vector represents a displacement in the direction perpendicular to the midline

                v1 = midline + dxy
                v2 = midline - dxy
                # calculate two points, v1 and v2, located at distances i units away from the midline in opposite directions.

                p1 = im_interp2d.ev(v1.T[0], v1.T[1])
                ##calculate a value p1 by evaluating the image data at the coordinates v1

                p2 = im_interp2d.ev(v2.T[0], v2.T[1])

                data = np.vstack([p1, data, p2])
                # Stack arrays in sequence vertically (row wise).

            # Calculate the average profile
            prf = np.average(data, axis=0)
            sigma = 2  # standard deviation of Gaussian filter

            prf = gaussian_filter1d(prf, sigma)
            # A Gaussian Filter is a low pass filter used for reducing noise (high frequency components)
            # and blurring regions of an image.
            prf_list.append(prf)
        else:
            prf_list.append([])
    return prf_list


# calculates the constriction degree in relative and absolute value,
# and the relative position along the length of the nucleoid
# relative constr degree needs to divided by the height (hgt) of the local minima
def constr_degree(intensities, new_step_lengths, width_not_ordered):
    constrDegree_list = []
    relPos_list = []
    constrDegree_abs_list = []
    ctpos_list = []

    for intensity, width_no, new_step_length in zip(
        intensities, width_not_ordered, new_step_lengths
    ):
        if not np.any(intensity):
            fill_nan_values(
                constrDegree_list, constrDegree_abs_list, ctpos_list, relPos_list
            )
        else:
            weighted_intensity = intensity * width_no
            minima = get_minima(weighted_intensity)
            if all(not x for x in minima) or np.sum(weighted_intensity.T) == 0:
                minsize, minsizeabs, ctpos, ctposr = 0, 0, np.nan, np.nan
            else:
                index, dh, dhi, hgt = identify_minima(weighted_intensity, minima)
                fix = np.argmax(dh)
                minsizeabs = dhi[fix]
                minsize = minsizeabs / hgt[fix]
                ctpos = index[fix]
                ctposr = np.cumsum(new_step_length)[ctpos] / np.sum(new_step_length)

            assign_values(
                constrDegree_list,
                constrDegree_abs_list,
                ctpos_list,
                relPos_list,
                minsize,
                minsizeabs,
                ctpos,
                ctposr,
            )

    avg_consDegree = [np.nan] if not constrDegree_list else np.mean(constrDegree_list)
    avg_abs_consDegree = (
        [np.nan] if not constrDegree_abs_list else np.mean(constrDegree_abs_list)
    )
    return (
        constrDegree_list,
        relPos_list,
        constrDegree_abs_list,
        ctpos_list,
        avg_consDegree,
        avg_abs_consDegree,
    )


def fill_nan_values(constrDegree_list, constrDegree_abs_list, ctpos_list, relPos_list):
    for _ in range(4):
        constrDegree_list.append(np.nan)
    ctpos_list.append(np.nan)
    relPos_list.append(np.nan)


def get_minima(weighted_intensity):
    return np.concatenate(
        (
            [False],
            (weighted_intensity.T[1:-1] < weighted_intensity.T[:-2])
            & (weighted_intensity.T[1:-1] < weighted_intensity.T[2:]),
            [False],
        )
    )


def identify_minima(intensity, minima):
    index = np.where(minima)[0]
    dh, dhi, hgt = np.zeros(index.shape), np.zeros(index.shape), np.zeros(index.shape)
    for i, k in enumerate(index):
        half1 = intensity.T[: k - 1]
        half2 = intensity.T[k + 1 :]
        try:
            dh1 = np.max(half1) - intensity.T[k]
            dh2 = np.max(half2) - intensity.T[k]
            dh[i] = np.min([dh1, dh2])
            dhi[i] = np.mean([dh1, dh2])
            hgt[i] = intensity.T[k] + dhi[i]
        except ValueError:
            pass
    return index, dh, dhi, hgt


def assign_values(
    constrDegree_list,
    constrDegree_abs_list,
    ctpos_list,
    relPos_list,
    minsize,
    minsizeabs,
    ctpos,
    ctposr,
):
    constrDegree_list.append(minsize)
    constrDegree_abs_list.append(minsizeabs)
    ctpos_list.append(ctpos)
    relPos_list.append(ctposr)


# calculate the average width of segments defined by coordinates (x1, y1) and (x2, y2)
def get_obj_avg_width(x1s, y1s, x2s, y2s, px):
    width_not_ordered_list = []
    width_list = []

    for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
        if np.any(x1):
            width_not_ordered = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * px
            width_not_ordered_list.append(width_not_ordered)

            sorted_width = sorted(width_not_ordered, reverse=True)
            width = sum(sorted_width[: math.floor(len(sorted_width) / 3)]) / math.floor(
                len(sorted_width) / 3
            )
            width_list.append(width)
        else:
            width_not_ordered_list.append(np.nan)
            width_list.append(np.nan)  # Assign np.nan for empty lists

    average_width = (
        np.nan if not width_list else np.nanmean(width_list)
    )  # Calculate average handling NaNs
    return width_list, width_not_ordered_list, average_width


def get_nucleoid_volume(x1s, y1s, x2s, y2s, step_lengths, px):

    nucleoid_volume_list = []

    for x1, y1, x2, y2, step_length in zip(x1s, y1s, x2s, y2s, step_lengths):
        if np.any(x1):
            d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            volume = np.trapz((np.pi * (d / 2) ** 2), dx=step_length)
            nucleoid_volume_list.append(volume * px * px)

    total_volume = np.nan if not nucleoid_volume_list else np.sum(nucleoid_volume_list)
    return nucleoid_volume_list, total_volume


def get_object_step_length(x1s, y1s, x2s, y2s, px):

    nuc_step_length_list = []
    nuc_length_list = []
    for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
        if np.any(x1):

            # calculated as the difference between corresponding x-coordinates of two line segments
            dx = x1[1:] + x2[1:] - x1[:-1] - x2[:-1]
            # x1[1:] => start from index 1

            dy = y1[1:] + y2[1:] - y1[:-1] - y2[:-1]

            # Calculate the step length for each segment and multiply by pixel size
            step_length = (np.sqrt(dx**2 + dy**2) / 2) * px
            nuc_step_length_list.append(step_length)
            nuc_length_list.append(np.sum(step_length))
        else:
            nuc_step_length_list.append([])
            nuc_length_list.append([])
    return nuc_step_length_list, nuc_length_list


def get_object_surface_area(width_nos, step_lengths):

    sum_surface_area_list = []

    for width_no, step_length in zip(width_nos, step_lengths):
        if not np.any(np.isnan(width_no)):

            widths = width_no[1:]

            # Calculate the surface areas for each segment
            surface_areas = 2 * np.pi * (widths / 2) * step_length

            # Calculate the total surface area by summing up individual surface areas
            sum_surface_area = np.sum(surface_areas)
            sum_surface_area_list.append(sum_surface_area)
        else:
            sum_surface_area_list.append(np.nan)
        total_surface_area = (
            [np.nan] if not sum_surface_area_list else np.mean(sum_surface_area_list)
        )

    return sum_surface_area_list, total_surface_area


def get_object_surface_area_over_volume(surface_areas, volumes):

    ratio_list = []

    for surface_area, volume in zip(surface_areas, volumes):

        ratio = surface_area / volume
        ratio_list.append(ratio)

    avg_ratio = [np.nan] if not ratio_list else np.mean(ratio_list)

    return ratio_list, avg_ratio


def object_bending_energy(data, px):
    """
    Simplification of the formula from https://doi.org/10.1016/j.comgeo.2013.09.001
    """
    bending_energy_list = []
    curvatures, _, _, _, _, _, _ = get_object_curvature_characteristics(data, px)
    for line, curvature in zip(data, curvatures):
        BE = bending_energy(line, curvature, px)
        bending_energy_list.append(np.round(BE, 5))

    avg_bending_energy = (
        [np.nan] if not bending_energy_list else np.mean(bending_energy_list)
    )
    return bending_energy_list, avg_bending_energy


def bending_energy(line, curvature, px):
    length = line_length(line) * px
    sum_squared_curvature = np.sum(curvature**2)
    bending_energy = sum_squared_curvature / length
    return np.round(bending_energy, 5)


def midline_midpoint(midline):
    middle_index = len(midline) // 2
    middle_point = midline[middle_index]
    return middle_point


def get_object_midpoint(midlines):
    centroid_coordinates = []
    for midline in midlines:
        middle_point = midline_midpoint(midline)
        centroid_coordinates.append(middle_point)
    return centroid_coordinates


def get_centroid(cropped_masks):

    labels = label(cropped_masks)
    props = regionprops_table(labels, properties=("label", "centroid"))
    centroids = [props["centroid-0"], props["centroid-1"]]

    return [centroids[0][0], centroids[1][0]]


def get_object_centroid(cropped_masks):

    centroid_list = []

    for nucleoid_mask in cropped_masks:
        centroid = get_centroid(nucleoid_mask)
        centroid_list.append(centroid)

    return centroid_list


def get_interobject_distance(centroid_coordinates, px):
    num_pairs = len(list(combinations(centroid_coordinates, 2)))

    if num_pairs == 0:
        return np.nan, np.nan  # No pairs found

    distances = [distance(c1, c2) for c1, c2 in combinations(centroid_coordinates, 2)]
    total_distance = sum(distances) * px
    average_distance = total_distance / num_pairs

    return average_distance, distances


def get_object_sinuosity(midlines):

    sinuosity_list = []

    for midline in midlines:
        temp_sinuosity = sinuosity(midline)
        sinuosity_list.append(temp_sinuosity)
    avg_sinuosity = [np.nan] if not sinuosity_list else np.mean(sinuosity_list)

    return sinuosity_list, avg_sinuosity


def get_mesh_area(width_no, step_length):
    result = [step * width for step, width in zip(step_length, width_no)]
    mesh_area = sum(result)
    return mesh_area


def ratio_mesh_over_total_area(step_length, width_no, areas):
    ratio_list = []
    for step_length, width_no, area in zip(step_length, width_no, areas):
        mesh_area = get_mesh_area(width_no, step_length)
        ratio = mesh_area / area
        ratio_list.append(ratio)
    return ratio_list


def get_shortest_pole_object_distances(pole_coordinates, centroid_coordinates, px):
    if len(centroid_coordinates) == 0:
        return np.nan, np.nan  # No centroid coordinates found

    distances = [
        distance(pole, c) for pole in pole_coordinates for c in centroid_coordinates
    ]
    average_distance = np.mean(distances) * px if distances else np.nan

    return average_distance, distances


def get_pole_coordinates(mesh):
    pole1 = [mesh[0, 0], mesh[0, 1]]
    pole2 = [mesh[-1, 0], mesh[-1, 1]]
    return pole1, pole2


def get_pole_object_distances(mesh, centroid_coordinates, px):
    if len(centroid_coordinates) == 0:
        return np.nan, np.nan  # No centroid coordinates found
    pole_coordinates = get_pole_coordinates(mesh)
    distances = [
        distance(pole, c) for pole in pole_coordinates for c in centroid_coordinates
    ]
    avg_distance = np.mean(distances) * px if distances else np.nan
    # Sort the distances
    sorted_distances = np.sort(distances)
    # Take the two smallest distances
    min_2_distances = sorted_distances[:2] * px
    return avg_distance, min_2_distances, distances


def angle_between_vector(v1, v2):
    d1 = np.sqrt(np.sum(v1**2))
    d2 = np.sqrt(np.sum(v2**2))
    return np.arccos(np.dot(v1, v2) / (d1 * d2))


def projection_on_midline(centroids, midline, width, px):

    L = line_length(midline)
    sorted_width = sorted(width, reverse=True)
    avg_width = sum(sorted_width[: math.floor(len(sorted_width) / 3)]) / math.floor(
        len(sorted_width) / 3
    )
    l_list = []
    dl_list = []
    norm_l_list = []
    norm_dl_list = []
    for pp in centroids:
        v_p = np.array(pp)[np.newaxis, :] - midline
        v_p_length = np.sqrt(np.sum(v_p**2, axis=1))

        nearest_loc = np.argmin(v_p_length)
        if nearest_loc == len(midline) - 1:
            v_m = midline[-1] - midline[-2]
        else:
            v_m = midline[nearest_loc + 1] - midline[nearest_loc]

        dl = np.dot(v_p[nearest_loc], np.array([-v_m[1], v_m[0]])) / np.linalg.norm(v_m)

        # Determine the sign of dl based on the orientation of the midline
        if np.cross(v_m, v_p[nearest_loc]) < 0:
            dl = -dl

        # Normalize signed distance
        norm_dl = dl / avg_width

        v_x_proj = np.array([1, 0])
        ang1 = angle_between_vector(v_m, v_x_proj)
        ang2 = angle_between_vector(v_p[nearest_loc], v_x_proj)
        ang12 = ang1 - ang2
        dx = np.cos(ang12) * dl
        dy = np.sin(ang12) * dl

        if nearest_loc == 0:
            l = 0
        else:
            l = line_length(midline[: nearest_loc + 1])

        w = width[nearest_loc]

        if w == 0:
            norm_dy = dy / (2 * np.max(width))
            norm_dx = dx / L
        else:
            norm_dx, norm_dy = dx / L, dy / w
        norm_l = l / L

        l_list.append(np.round(l, 5) * px)
        dl_list.append(np.round(dl, 5) * px)
        norm_l_list.append(np.round(norm_l, 5))
        norm_dl_list.append(np.round(norm_dl, 5))

    return l_list, dl_list, norm_l_list, norm_dl_list


def get_avg_distance_from_center(l_norm_list):
    average_pos = np.mean(l_norm_list)
    avg_dist = np.abs(average_pos - 0.5)
    return avg_dist


def get_object_intensities(object_masks, cropped_signal):
    individual_intensity_values = []

    # Loop through each mask to get the intensity values
    for mask in object_masks:
        intensity_values = cropped_signal[mask != 0]
        individual_intensity_values.append(intensity_values)

    # Flatten the list of individual intensities to get the combined intensities
    combined_intensity_values = np.concatenate(individual_intensity_values)

    return combined_intensity_values, individual_intensity_values


def get_objects_mean(intensity_values_list):
    means = [np.mean(intensity_values) for intensity_values in intensity_values_list]
    return means


def get_objects_std(intensity_values_list):
    stds = [np.std(intensity_values) for intensity_values in intensity_values_list]
    return stds


def get_objects_kurtosis(intensity_values_list):
    kurtoses = [
        kurtosis(intensity_values) for intensity_values in intensity_values_list
    ]
    return kurtoses


def get_objects_skewness(intensity_values_list):
    skewnesses = [skew(intensity_values) for intensity_values in intensity_values_list]
    return skewnesses


def get_variance(data):
    return np.var(data)


# %% Profiling
def measure_smoothened_intensity(midline, im_interp2d, width=7, subpixel=0.5):

    data = measure_intensity_interp2d(
        midline, im_interp2d, width=width, subpixel=subpixel
    )
    prf = gaussian_smoothing(data)
    return prf


def gaussian_smoothing(data):
    prf = np.average(data, axis=0)
    sigma = 2  # standard deviation of Gaussian filter
    prf = gaussian_filter1d(prf, sigma)
    return prf.flatten()


def measure_intensity_interp2d(midline, im_interp2d, width=7, subpixel=0.5):

    unit_dxy = unit_perpendicular_vector(midline, closed=False)
    width_normalized_dxy = unit_dxy * subpixel

    data = im_interp2d.ev(midline.T[0], midline.T[1])
    for i in range(1, 1 + int(width * 0.5 / subpixel)):
        dxy = width_normalized_dxy * i
        v1 = midline + dxy
        v2 = midline - dxy
        p1 = im_interp2d.ev(v1.T[0], v1.T[1])
        p2 = im_interp2d.ev(v2.T[0], v2.T[1])
        data = np.vstack([p1, data, p2])
    return data


def normalize_per_cell(profile):
    """
    Normalize the values in a 1D profile using Min-Max scaling.

    Parameters:
        profile (array-like): A 1D array-like object representing the profile.

    Returns:
        normalized_profile (ndarray): Normalized profile.
    """
    # Convert the input to a numpy array for ease of manipulation
    profile = np.array(profile)

    # Calculate the minimum and maximum values in the profile
    min_val = np.min(profile)
    max_val = np.max(profile)

    # Check if min_val equals max_val to avoid division by zero
    if min_val == max_val:
        return np.zeros_like(
            profile
        )  # Return an array of zeros if all values are the same

    # Normalize each value to the range [0, 1]
    normalized_profile = (profile - min_val) / (max_val - min_val)

    return normalized_profile


def get_cumulative_step_length_midline(midline, px):
    """
    Calculate cumulative step lengths for each segment in the midline.

    Parameters:
        midline (ndarray): An array containing X and Y coordinates of the midline.
        px (float): The pixel size.

    Returns:
        cumulative_step_lengths (ndarray): Array containing cumulative step lengths for each segment in the midline.
    """
    # Extract X and Y coordinates from the midline
    x = midline[:, 0]
    y = midline[:, 1]

    # Compute the step lengths
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    step_lengths = np.sqrt(dx**2 + dy**2) * px

    # Calculate cumulative step lengths
    cumulative_step_lengths = np.zeros_like(x)
    cumulative_step_lengths[1:] = np.cumsum(step_lengths)

    return cumulative_step_lengths


def linear_interpolation(array, num_values):
    # Original x-coordinates
    x_original = np.arange(len(array))
    # New x-coordinates for interpolation
    x_new = np.linspace(0, len(array) - 1, num_values)
    # Perform linear interpolation
    interpolated_values = np.interp(x_new, x_original, array)
    return interpolated_values


# %% Contour shift functions
def get_phase_mask(phase_cropped_img, phase_log_sigma=0.5):

    inverted_phase_cropped_image = cv.bitwise_not(phase_cropped_img)

    log_filtered_img_phase = gaussian_laplace(
        inverted_phase_cropped_image, sigma=phase_log_sigma
    )
    threshold_value = threshold_otsu(log_filtered_img_phase)
    cropped_phase_mask = log_filtered_img_phase > threshold_value
    cropped_phase_mask = cropped_phase_mask.astype(np.uint8)

    return cropped_phase_mask


def closing_mask(cropped_phase_mask, closing_level=2):

    mask_closed = closing(
        cropped_phase_mask, footprint=np.ones((closing_level, closing_level))
    )

    return mask_closed


def find_centroid_in_largest_mask(phase_mask_closed, phase_cropped_contour):
    # Find the largest connected component in the phase mask
    labeled_mask, num_labels = label(phase_mask_closed)

    if num_labels > 0:
        label_sizes = ndimage.sum(
            phase_mask_closed, labeled_mask, range(1, num_labels + 1)
        )
        largest_label = np.argmax(label_sizes) + 1  # +1 to adjust for 0-based indexing
        largest_mask = labeled_mask == largest_label

        # Get the centroid of the largest mask
        phase_centroid = get_centroid(largest_mask)

        # Check if the centroid lies within the cell contour
        phase_cropped_contour = phase_cropped_contour.astype(np.float32)
        if cv.pointPolygonTest(phase_cropped_contour, phase_centroid, False) >= 0:
            return [phase_centroid]

    return []


def align(signal_centroid, phase_centroid, cropped_contour):

    # Calculate the translation vector
    translation_vector = np.array(signal_centroid) - np.array(phase_centroid)

    # Ensure translation_vector is a 1D array
    translation_vector = translation_vector.flatten()
    aligned_contour = cropped_contour.copy()

    # colume of contour or midline is 2
    if aligned_contour.shape == (len(aligned_contour), 2):
        aligned_contour = aligned_contour + translation_vector

    # colume of mesh is 4
    if aligned_contour.shape == (len(aligned_contour), 4):
        # Add translation vector to the first and second columns
        aligned_contour[:, :2] += translation_vector

        # Add translation vector to the third and fourth columns
        aligned_contour[:, 2:] += translation_vector

    return aligned_contour


def align_objects(
    signal_centroid, phase_centroid, cropped_mesh, cropped_contour, cropped_midline
):
    # Aligned contour
    aligned_contour = align(signal_centroid, phase_centroid, cropped_contour)

    aligned_mesh = align(signal_centroid, phase_centroid, cropped_mesh)

    aligned_midline = align(signal_centroid, phase_centroid, cropped_midline)

    return aligned_contour, aligned_midline, aligned_mesh


def get_signal_mask(signal, contour, log_sigma, min_overlap_ratio, max_external_ratio):

    cropped_img, _, cropped_contour, x_offset, y_offset = crop_image(
        image=signal, contour=contour, mask_to_crop=None
    )
    cropped_mask = draw_mask(cropped_contour, cropped_img)
    bgr = mean_background(cropped_img, cropped_mask)
    cropped_img_bgr = np.maximum(cropped_img - bgr, 0)
    # Apply Laplacian of Gaussian (LoG) filter to the cropped_img
    log_filtered_img = gaussian_laplace(cropped_img_bgr, sigma=log_sigma)

    threshold_value = threshold_otsu(log_filtered_img)

    object_mask = log_filtered_img < threshold_value

    object_mask = keep_significant_masks(
        object_mask, cropped_mask, min_overlap_ratio, max_external_ratio
    )

    return object_mask, cropped_img, x_offset, y_offset


def shift_contour(
    phase_image,
    signal_image,
    contour,
    mesh,
    midline,
    log_sigma=1.5,
    kernel_width=3,
    min_overlap_ratio=0.3,
    max_external_ratio=0.8,
    phase_log_sigma=0.5,
    phase_closing_level=2,
    signal_closing_level=12,
    max_shift_correction=10,
):
    """
    Aligns the contour, mesh, and midline based on the centroids of the phase and signal masks.

    Parameters:
    - phase_image (array): The phase image.
    - signal_image (array): The signal image.
    - contour (array): The contour data.
    - mesh (array): The mesh data.
    - midline (array): The midline data.
    - log_sigma (float): Sigma for the log filter for the signal mask.
    - kernel_width (int): Width of the dilation kernel.
    - min_overlap_ratio (float): Minimum overlap ratio for the signal mask.
    - max_external_ratio (float): Maximum external ratio for the signal mask.
    - phase_log_sigma (float): Sigma for the log filter for the phase mask.
    - phase_closing_level (int): Closing level for the phase mask.
    - signal_closing_level (int): Closing level for the signal mask.
    - max_shift_correction (float): Maximum allowable distance between centroids for correction.

    Returns:
    - aligned_contour (array): The aligned contour data.
    - aligned_mesh (array): The aligned mesh data.
    - aligned_midline (array): The aligned midline data.
    """
    # Phase image
    phase_cropped_img, _, phase_cropped_contour, x, y = crop_image(phase_image, contour)

    # FM4-64 image
    signal_mask, cropped_signal, _, _ = get_signal_mask(
        signal_image, contour, log_sigma, min_overlap_ratio, max_external_ratio
    )

    cropped_contour, cropped_midline, cropped_mesh = get_cropped_cell_data(
        contour, midline, mesh, x, y
    )

    # Get phase mask
    cropped_phase_mask = get_phase_mask(
        phase_cropped_img, phase_log_sigma=phase_log_sigma
    )

    # Fill the hole within mask
    phase_mask_closed = closing_mask(
        cropped_phase_mask, closing_level=phase_closing_level
    )
    signal_mask_closed = closing_mask(signal_mask, closing_level=signal_closing_level)

    kernel = np.ones(
        (kernel_width, kernel_width), np.uint8
    )  # You can adjust the kernel size as needed

    phase_mask_closed = cv.dilate(
        phase_mask_closed.astype(np.uint8), kernel, iterations=2
    )
    signal_mask_closed = cv.dilate(
        signal_mask_closed.astype(np.uint8), kernel, iterations=2
    )

    # Find largest mask
    phase_centroid = find_centroid_in_largest_mask(
        phase_mask_closed, phase_cropped_contour
    )
    signal_centroid = get_centroid(signal_mask_closed)

    # Calculate the distance between the centroids
    distance = np.linalg.norm(np.array(phase_centroid) - np.array(signal_centroid))

    # Check if the distance exceeds the maximum allowable shift correction
    if distance > max_shift_correction:
        return None, None, None

    cropped_aligned_contour, cropped_aligned_midline, cropped_aligned_mesh = (
        align_objects(
            signal_centroid,
            phase_centroid,
            cropped_mesh,
            cropped_contour,
            cropped_midline,
        )
    )
    aligned_contour, aligned_midline, aligned_mesh = get_uncropped_cell_data(
        cropped_aligned_contour, cropped_aligned_midline, cropped_aligned_mesh, x, y
    )

    return aligned_contour, aligned_mesh, aligned_midline


def crop_signal_with_optional_shift(image_obj, channel, contour, shift_signal):
    phase_img = image_obj.image if shift_signal else None
    cropped_signal, cropped_mask, cropped_contour, x_offset, y_offset = crop_image(
        image_obj.bg_channels[channel], contour, phase_img=phase_img
    )
    return cropped_signal, cropped_mask, cropped_contour, x_offset, y_offset


# %% Membrane features


def change_contour_array_length(cell_perimeter, contour, scale=0.02, smooth=1):
    num = int(cell_perimeter / scale)
    adjust_eroded_contour = spline_approximation(
        contour, n=num, smooth_factor=smooth, closed=True
    )
    return adjust_eroded_contour

def normalized_intensity_by_image(profile, cropped_image):
    # Convert the input to a numpy array for ease of manipulation
    prof = np.array(profile)

    # Flatten the arrays to avoid issues with multidimensional input
    prof_flat = prof.flatten()
    cropped_image_flat = np.array(cropped_image).flatten()

    # Calculate the mean of the lowest 5% values in the profile
    min_percentile_value = np.percentile(cropped_image_flat, 5)
    max_percentile_value = np.percentile(prof_flat, 95)

    # Calculate the mean of the lowest 5% values in the cropped image
    min_intensity = np.mean(cropped_image_flat[cropped_image_flat <= min_percentile_value])
    # Calculate the mean of the highest 5% values in the profile
    max_intensity = np.mean(prof_flat[prof_flat >= max_percentile_value])

    # Check if min_intensity equals max_intensity to avoid division by zero
    if np.isclose(max_intensity, min_intensity):
        return np.zeros_like(prof)  # Return an array of zeros if all values are the same

    # Normalize each value to the range [0, 1] using the calculated min and max intensities
    normalized_profile = (prof - min_intensity) / (max_intensity - min_intensity)

    return normalized_profile

def get_homogeneity(normalized_intensity):
    # for normalized_int in normalized_intensity:
    gradient = np.gradient(normalized_intensity)
    homogeneity = abs(np.median(abs(gradient)) - np.mean(abs(gradient)))
    return homogeneity


def set_bg_zero(contour, cropped_img):
    cell_cropped_mask = draw_mask(contour, cropped_img)
    bg_zero_img = cropped_img * cell_cropped_mask
    return bg_zero_img


def rescale_image_255(cropped_image, max_value=255):

    # Find the minimum and maximum intensity values
    min_intensity = np.min(cropped_image)
    max_intensity = np.max(cropped_image)

    # Scale the intensity values to the range [0, 255]
    scaled_image = (
        (cropped_image - min_intensity) / (max_intensity - min_intensity) * max_value
    )

    # Float images are not supported by graycomatrix. Convert the image to an unsigned integer type.
    scaled_image = scaled_image.astype(np.uint16)

    return scaled_image


def create_co_occurrence_matrix(
    cropped_img,
    distances=[1],
    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    levels=256,
    symmetric=True,
    normed=True,
):
    """
    Create a co-occurrence matrix from a cropped image.

    Parameters:
    - cropped_image: The cropped grayscale image.
    - distances: List of pixel pair distances.
    - angles: List of angles in radians for pixel pairs.
    - levels: Number of gray levels (typically 256 for 8-bit images).
    - symmetric: If True, the output matrix P[:, :, d, theta] is symmetric.
                 This is accomplished by ignoring the order of value pairs, so both (i, j) and (j, i) are accumulated when (i, j) is encountered for a given offset.
    - normed: If True, normalize each matrix P[:, :, d, theta] by dividing by the total number of accumulated co-occurrences for the given offset.

    Returns:
    - co_occurrence_matrix: The co-occurrence matrix.
    """
    # Compute the co-occurrence matrix
    co_occurrence_matrix = graycomatrix(
        cropped_img,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    co_occurrence_matrix[:, :, :, :] *= np.where(
        (np.arange(levels) == 0)[:, None, None, None]
        | (np.arange(levels) == 0)[None, :, None, None],
        0,
        1,
    )

    return co_occurrence_matrix


def calculate_haralick_features(
    contour,
    cropped_img,
    distances=[2],
    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    levels=256,
):
    bg_zero_img = set_bg_zero(contour, cropped_img)
    bg_zero_img_rescaled = rescale_image_255(bg_zero_img, max_value=255)
    glcm = create_co_occurrence_matrix(
        bg_zero_img_rescaled,
        distances=distances,
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    glcm_dissimilarity = graycoprops(glcm, "dissimilarity").mean()
    glcm_correlation = graycoprops(glcm, "correlation").mean()
    glcm_homogeneity = graycoprops(glcm, "homogeneity").mean()
    glcm_energy = graycoprops(glcm, "energy").mean()
    glcm_contrast = graycoprops(glcm, "contrast").mean()
    glcm_entropy = shannon_entropy(glcm, base=2)

    return (
        glcm_dissimilarity,
        glcm_correlation,
        glcm_homogeneity,
        glcm_energy,
        glcm_contrast,
        glcm_entropy,
    )


def calculate_object_haralick_features(
    object_masks,
    cropped_img,
    distances=[2],
    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    levels=256,
):
    combined_mask = np.sum(object_masks, axis=0)
    bg_zero_img = cropped_img * combined_mask
    bg_zero_img_rescaled = rescale_image_255(bg_zero_img, max_value=255)
    glcm = create_co_occurrence_matrix(
        bg_zero_img_rescaled,
        distances=distances,
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    glcm_dissimilarity = graycoprops(glcm, "dissimilarity").mean()
    glcm_correlation = graycoprops(glcm, "correlation").mean()
    glcm_homogeneity = graycoprops(glcm, "homogeneity").mean()
    glcm_energy = graycoprops(glcm, "energy").mean()
    glcm_contrast = graycoprops(glcm, "contrast").mean()
    glcm_entropy = shannon_entropy(glcm, base=2)
    return (
        glcm_dissimilarity,
        glcm_correlation,
        glcm_homogeneity,
        glcm_energy,
        glcm_contrast,
        glcm_entropy,
    )


def calculate_contour_haralick_features(
    contour_intensities,
    distances=[2],
    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    levels=256,
):
    rescaled_contour_intensities = rescale_image_255(contour_intensities, max_value=255)
    glcm = create_co_occurrence_matrix(
        rescaled_contour_intensities,
        distances=distances,
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    glcm_dissimilarity = graycoprops(glcm, "dissimilarity").mean()
    glcm_correlation = graycoprops(glcm, "correlation").mean()
    glcm_homogeneity = graycoprops(glcm, "homogeneity").mean()
    glcm_energy = graycoprops(glcm, "energy").mean()
    glcm_contrast = graycoprops(glcm, "contrast").mean()
    glcm_entropy = shannon_entropy(glcm, base=2)

    return (
        glcm_dissimilarity,
        glcm_correlation,
        glcm_homogeneity,
        glcm_energy,
        glcm_contrast,
        glcm_entropy,
    )


# Function to split the row's array into two halves
def split_array(row):
    midpoint = len(row) // 2
    first_half = row[:midpoint]
    second_half = row[midpoint:]
    return first_half, second_half


def get_complemented_contour_intensity(row):

    first_half, second_half = split_array(row)

    # Invert the 'second_half' array
    inverted_second_half = second_half[::-1]

    # Add corresponding elements from 'first_half' and inverted 'second_half' arrays
    complemented_contour_intensity_profile = np.array(
        [(a + b) / 2 for a, b in zip(first_half, inverted_second_half)]
    )

    return complemented_contour_intensity_profile


# %% Shift functions
def shift_image(img, shift):
    """
    Correct XY drift between phase contrast image and fluorescent image(s).
    :param img: Input image
    :param shift: Subpixel XY drift
    :return: Drift corrected image
    """
    offset_image = fourier_shift(np.fft.fftn(img), shift)
    offset_image = np.fft.ifftn(offset_image)
    offset_image = np.round(offset_image.real)
    offset_image[offset_image <= 0] = 10

    return offset_image.astype(np.uint16)
