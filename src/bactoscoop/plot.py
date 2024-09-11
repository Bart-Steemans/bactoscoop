# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:02:27 2023

@author: Bart Steemans. Govers Lab.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import ListedColormap
from skimage.color import label2rgb
from skimage.measure import label
from . import utilities as u

dpi = 300


def plot_contour(cell_id, image, verbose=False):
    """
    Plot the contour of a cell on an image.

    This function takes a cell object and an image object and plots the contour of the specified cell on the image.
    It also displays the cell ID in the title of the plot.

    Parameters
    ----------
    cell : Cell
        The cell object to be plotted.
    image : Image
        The image object on which to plot the cell contour.
    """
    cellobj = next((cell for cell in image.cells if cell.cell_id == cell_id), None)
    if cellobj is None:
        if verbose:
            print(f"Cell with ID {cell_id} not found.")
        return
    fig = plt.figure(figsize=(12, 12))

    contour = cellobj.contour
    midline = cellobj.midline
    msh = cellobj.mesh
    xmin = np.min(contour.T[1] - 10)
    xmax = np.max(contour.T[1] + 10)
    ymin = np.min(contour.T[0] - 10)
    ymax = np.max(contour.T[0] + 10)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.imshow(image.image, cmap="gist_gray")
    plt.plot(contour.T[1], contour.T[0], "-", c="w", lw=5)
    # plt.plot(midline.T[1], midline.T[0], '-', c='blue', lw=5)
    # plt.plot([msh[:,1][::2], msh[:,3][::2]], [msh[:,0][::2], msh[:,2][::2]], '-', c='cyan', lw=2)
    plt.title(f"Cell {cellobj.cell_id} Frame {image.frame}")
    plt.show()


def plot_aligned_contours(cell_id, image, channel, verbose=False):
    """
    Plot the contour of a cell on an image.

    This function takes a cell object and an image object and plots the contour of the specified cell on the image.
    It also displays the cell ID in the title of the plot.

    Parameters
    ----------
    cell : Cell
        The cell object to be plotted.
    image : Image
        The image object on which to plot the cell contour.
    """
    cellobj = next((cell for cell in image.cells if cell.cell_id == cell_id), None)
    if cellobj is None:
        if verbose:
            print(f"Cell with ID {cell_id} not found.")
        return
    fig = plt.figure(figsize=(12, 12))

    contour = cellobj.contour
    midline = cellobj.midline
    aligned_contour = cellobj.shifted_contour
    xmin = np.min(contour.T[1] - 10)
    xmax = np.max(contour.T[1] + 10)
    ymin = np.min(contour.T[0] - 10)
    ymax = np.max(contour.T[0] + 10)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.imshow(image.channels[channel], cmap="gist_gray")
    plt.plot(contour.T[1], contour.T[0], "-", c="w", lw=5)
    if aligned_contour is not None:
        plt.plot(aligned_contour.T[1], aligned_contour.T[0], "-", c="orange", lw=5)
    # plt.plot(midline.T[1], midline.T[0], '-', c='blue', lw=5)

    plt.title(f"Cell {cellobj.cell_id} Frame {image.frame}")
    plt.show()


def plot_svm_controls(frame_cell_pairs, image_objects, message=None):
    """
    Plot cells based on frame and cell ID pairs from SVM control analysis.

    This function takes frame and cell ID pairs, along with a list of image objects, and plots the cells from SVM control analysis.
    It allows adding an optional message to the title of the plot.

    Parameters
    ----------
    frame_cell_pairs : list of tuples
        A list of (frame, cell_id) pairs for cells to be plotted.
    image_objects : list
        A list of image objects corresponding to the frames in frame_cell_pairs.
    message : str, optional
        An optional message to be added to the title of the plot.
    """
    for cell_id, frame in frame_cell_pairs:
        image_obj = image_objects[frame]
        cellobj = next(
            (cell for cell in image_obj.cells if cell.cell_id == cell_id), None
        )
        contour = cellobj.contour
        # ori_contour = image.mesh_dataframe['contour'][cell]
        xmin = np.min(contour.T[1] - 10)
        xmax = np.max(contour.T[1] + 10)
        ymin = np.min(contour.T[0] - 10)
        ymax = np.max(contour.T[0] + 10)

        plt.figure(figsize=(12, 12))
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.imshow(image_obj.image, cmap="gist_gray")
        plt.plot(contour.T[1], contour.T[0], "-", c="w", lw=5)
        if message is not None:
            plt.title(f"Cell: {cellobj.cell_id} ---Frame: {frame} ---Type: {message}")
            plt.show()


def plot_mask(image_object, joined=False, dpi=400, alpha=0.4):
    """
    Plot the mask overlay on an image.

    This function takes an image object and optionally a joined mask, and plots the mask overlay on the image.
    The alpha parameter controls the transparency of the mask overlay.

    Parameters
    ----------
    image_object : Image object
        An image object containing the image and mask data.
    joined : bool, optional
        Whether to use the joined mask for overlay (default is False).
    dpi : int, optional
        Dots per inch for the plot (default is 400).
    alpha : float, optional
        Alpha (transparency) value for the mask overlay (default is 0.4).
    """

    plt.figure(
        figsize=(image_object.image.shape[1] / dpi, image_object.image.shape[0] / dpi),
        dpi=dpi,
    )
    plt.imshow(image_object.image, cmap="gray")

    if joined is True:
        plt.imshow(
            label2rgb(label(image_object.joined_mask, connectivity=1), bg_label=0),
            alpha=alpha,
        )
    else:

        plt.imshow(
            label2rgb(label(image_object.mask, connectivity=1), bg_label=0), alpha=alpha
        )

    plt.axis("off")  # Turn off the axis
    plt.show()


def plot_random_contours(image, num_cells_to_plot=4, crop_size=200, scalebar=False):
    """
    Plot random cell contours within an image.

    This function selects and plots random cell contours within an image, showing a specified number of cells.

    Parameters
    ----------
    image : Image object
        An image object containing the image and cell contour data.
    num_cells_to_plot : int, optional
        Number of random cell contours to plot (default is 4).
    crop_size : int, optional
        Size of the cropped area around each cell (default is 200).
    scalebar : bool, optional
        Whether to add a scale bar to the plots (default is False).

    Returns
    -------
    None
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    plotted_cells = set()  # To keep track of which cells have been plotted

    for i in range(num_cells_to_plot):
        row = i // 2
        col = i % 2

        # Find a random cell that hasn't been plotted yet
        while True:
            cell_idx = random.randint(0, len(image.cells) - 1)
            if cell_idx not in plotted_cells:
                break

        cellobj = image.cells[cell_idx]
        contour = cellobj.contour
        center_x = np.mean(contour.T[1])
        center_y = np.mean(contour.T[0])
        half_crop_size = crop_size / 2
        xmin = center_x - half_crop_size
        xmax = center_x + half_crop_size
        ymin = center_y - half_crop_size
        ymax = center_y + half_crop_size

        axes[row, col].set_xlim(xmin, xmax)
        axes[row, col].set_ylim(ymin, ymax)
        axes[row, col].imshow(image.image, cmap="gist_gray")
        axes[row, col].plot(contour.T[1], contour.T[0], "-", c="w", lw=2)
        axes[row, col].set_title(f"Cell {cellobj.cell_id}")

        plotted_cells.add(cell_idx)  # Mark this cell as plotted
        if scalebar:
            scale_length_um = 1
            scale_length_px = scale_length_um / 0.065  # Convert µm to pixels
            scale_x = (
                xmin + (xmax - xmin) * 0.02
            )  # Adjust the X position of the scale bar
            scale_y = (
                ymin + (ymax - ymin) * 0.02
            )  # Adjust the Y position of the scale bar
            axes[row, col].plot(
                [scale_x, scale_x + scale_length_px],
                [scale_y, scale_y],
                color="white",
                lw=5,
            )

    # Remove axes for all subplots
    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    # plt.savefig('C:/Users/u0158103/Documents/PhD/Pictures/contour_gallery.svg', format='svg')
    plt.show()


def plot_random_objects(
    image, chann, num_objects_to_plot=4, crop_size=300, scalebar=True
):
    num_rows = 2  # Number of rows in the grid
    num_cols = 2  # Number of columns in the grid

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    plotted_objects = set()  # To keep track of which objects have been plotted

    for i in range(num_objects_to_plot):
        # Find a random cell that hasn't been plotted yet
        while True:
            object_idx = random.randint(0, len(image.cells) - 1)
            if object_idx not in plotted_objects:
                break

        cellobj = image.cells[object_idx]
        contour = cellobj.contour

        # Calculate cropping boundaries based on the fixed crop size
        center_x = np.mean(contour.T[1])
        center_y = np.mean(contour.T[0])
        half_crop_size = crop_size / 2
        xmin = center_x - half_crop_size
        xmax = center_x + half_crop_size
        ymin = center_y - half_crop_size
        ymax = center_y + half_crop_size

        row = i // num_cols
        col = i % num_cols

        axes[row, col].set_xlim(xmin, xmax)
        axes[row, col].set_ylim(ymin, ymax)
        axes[row, col].imshow(image.channels[chann], cmap="gist_gray")
        axes[row, col].plot(contour.T[1], contour.T[0], "-", c="yellow", lw=2)
        if cellobj.object_meshdata[chann]["object_contour"] is not None:
            for nuc_contour in cellobj.object_meshdata[chann]["object_contour"]:
                axes[row, col].plot(
                    nuc_contour.T[1], nuc_contour.T[0], "-", c="cyan", lw=2
                )

        axes[row, col].set_title(f"Cell {cellobj.cell_id}")
        axes[row, col].axis("off")

        # Add scale bar
        if scalebar:
            scale_length_um = 1
            scale_length_px = scale_length_um / 0.065  # Convert µm to pixels
            scale_x = (
                xmin + 10
            )  # Adjust the position of the scale bar (e.g., 10 pixels from the left)
            scale_y = (
                ymin + 10
            )  # Adjust the position of the scale bar (e.g., 10 pixels from the bottom)
            axes[row, col].plot(
                [scale_x, scale_x + scale_length_px],
                [scale_y, scale_y],
                color="white",
                lw=5,
            )

        plotted_objects.add(object_idx)  # Mark this object as plotted

    plt.tight_layout()
    plt.show()


def plot_random_axial(feature_dataframe, channels, method, num_objects_to_plot=2):
    num_plots = 2  # Number of vertical plots
    num_objects_per_plot = num_objects_to_plot // num_plots  # Objects per plot
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 10))  # Make the plot taller

    # Get a list of unique cell IDs
    unique_cell_ids = feature_dataframe["cell_id"].unique()

    # Randomly select cell IDs to plot
    random_cell_ids = random.sample(list(unique_cell_ids), num_objects_to_plot)

    for plot_index in range(num_plots):
        for i in range(num_objects_per_plot):
            cell_id = random_cell_ids[plot_index * num_objects_per_plot + i]

            for channel in channels:
                axial_intensity_column = f"{channel}_axial_intensity"
                cell_frame_condition = feature_dataframe["cell_id"] == cell_id

                if feature_dataframe[cell_frame_condition].empty:
                    continue

                axial_intensity = feature_dataframe[axial_intensity_column][
                    cell_frame_condition
                ].iloc[0]

                # Normalize the axial intensity curve by dividing by the maximum value
                normalized_midline_intensity_data = (
                    axial_intensity - np.min(axial_intensity)
                ) / (np.max(axial_intensity) - np.min(axial_intensity))

                color = f"C{channels.index(channel)}"  # Assign a unique color to each channel
                ax = axes[plot_index]

                ax.plot(
                    normalized_midline_intensity_data,
                    linestyle="-",
                    color=color,
                    label=f"Channel {channel}",
                )

            # Retrieve the corresponding frame for the cell
            frame = feature_dataframe["frame"][cell_frame_condition].iloc[0]

            ax = axes[plot_index]
            ax.set_title(f"Cell {cell_id}, Frame {frame}")
            ax.set_xlabel("Position along Axial")
            ax.set_ylabel("Normalized Axial Intensity")
            ax.set_ylim(0, 1.2)
            ax.legend(frameon=False)  # Remove the frame around the legend

    plt.tight_layout()
    plt.show()


def plot_normalized_axial_intensity(
    feature_dataframe, channels, selected_frame=None, selected_cell_id=None
):
    # Group the data by 'frame'
    grouped = feature_dataframe.groupby("frame")

    for frame, frame_group in grouped:
        # Check if a specific frame is selected, and if so, only proceed if it matches
        if selected_frame is not None and frame != selected_frame:
            continue

        for cell_id, group in frame_group.groupby("cell_id"):
            # Check if a specific cell_id is selected, and if so, only proceed if it matches
            if selected_cell_id is not None and cell_id != selected_cell_id:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            for channel in channels:
                axial_intensity = group[f"{channel}_axial_intensity"].iloc[
                    0
                ]  # Assuming all dataframes have the same length

                # Normalize the axial intensity curve by dividing by the maximum value
                normalized_midline_intensity_data = (
                    axial_intensity - np.min(axial_intensity)
                ) / (np.max(axial_intensity) - np.min(axial_intensity))

                color = f"C{channels.index(channel)}"  # Assign a unique color to each channel
                ax.plot(
                    normalized_midline_intensity_data,
                    linestyle="-",
                    color=color,
                    label=f"Channel {channel}",
                )

            ax.set_xlabel("Position along Axial")
            ax.set_ylabel("Normalized Axial Intensity")
            ax.set_title(
                f"Normalized Axial Intensity Plot for Frame {frame}, Cell {cell_id}"
            )
            ax.legend(frameon=False)  # Remove the frame around the legend

            # Set y-limit a bit higher than 1
            ax.set_ylim(0, 1.2)

            plt.show()


def plot_object_mesh(cell_id, image, chann, verbose=False):

    # Look up the cell by its ID
    cellobj = next((cell for cell in image.cells if cell.cell_id == cell_id), None)

    if cellobj is None:
        if verbose:
            print(f"Cell with ID {cell_id} not found.")
        return
    fig = plt.figure(figsize=(12, 12))
    contour = cellobj.contour
    xmin = np.min(contour.T[1] - 10)
    xmax = np.max(contour.T[1] + 10)
    ymin = np.min(contour.T[0] - 10)
    ymax = np.max(contour.T[0] + 10)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.imshow(image.channels[chann], cmap="gist_gray")
    plt.plot(contour.T[1], contour.T[0], "-", c="yellow", lw=2)

    if cellobj.object_meshdata[chann]["object_mesh"] is not None:
        for cnt, msh, mdl in zip(
            cellobj.object_meshdata[chann]["object_contour"],
            cellobj.object_meshdata[chann]["object_mesh"],
            cellobj.object_meshdata[chann]["object_midline"],
        ):
            if np.any(msh):
                plt.plot(
                    [msh[:, 1][::2], msh[:, 3][::2]],
                    [msh[:, 0][::2], msh[:, 2][::2]],
                    "-",
                    c="cyan",
                    lw=2,
                )
            plt.plot(cnt.T[1], cnt.T[0], "-", c="white", lw=2)

    plt.title(f"Cell {cellobj.cell_id} ----- Channel {chann} ----- Frame {image.frame}")
    plt.axis("off")
    plt.show()


def plot_object_contours(
    cell_id, image, contour_channels, shift=False, base_channel=None, verbose=False
):
    """
    Plot the contours of a cell and its objects for multiple channels on a single base channel image.

    Parameters:
    - cell_id (int): The ID of the cell to plot.
    - image: The image object containing cells and channels.
    - base_channel (str): The name of the base channel image to display.
    - contour_channels (list of str): List of channels for which to plot contours.
    - verbose (bool): If True, prints additional information if the cell is not found.
    """
    cellobj = next((cell for cell in image.cells if cell.cell_id == cell_id), None)

    if cellobj is None:
        if verbose:
            print(f"Cell with ID {cell_id} not found.")
        return

    fig = plt.figure(figsize=(12, 12))
    contour = cellobj.contour

    if base_channel is None:
        cropped_img, _, cropped_contour, x_offset, y_offset = u.crop_image(
            image=image.image, contour=contour, mask_to_crop=None, phase_img=None
        )
    elif shift:
        cropped_img, _, cropped_contour, x_offset, y_offset = u.crop_image(
            image=image.channels[base_channel],
            contour=contour,
            mask_to_crop=None,
            phase_img=image.image,
        )
    else:
        cropped_img, _, cropped_contour, x_offset, y_offset = u.crop_image(
            image=image.channels[base_channel],
            contour=contour,
            mask_to_crop=None,
            phase_img=None,
        )

    plt.imshow(cropped_img, cmap="gist_grey")

    plt.plot(cropped_contour.T[1], cropped_contour.T[0], color="yellow")
    for chann in contour_channels:
        if cellobj.object_meshdata[chann]["cropped_object_contour"] is not None:
            for obj_contour in cellobj.object_meshdata[chann]["cropped_object_contour"]:
                plt.plot(
                    obj_contour.T[1],
                    obj_contour.T[0],
                    "-",
                    lw=2,
                    label=f"Channel {chann}",
                )
                plt.legend()
    plt.title(f"Cell {cellobj.cell_id} ----- Base Channel {base_channel}")
    plt.axis("off")

    plt.show()


def plot_mesh_scientific(cell_id, image, crop_size=300, verbose=False):

    cellobj = next((cell for cell in image.cells if cell.cell_id == cell_id), None)

    if cellobj is None:
        if verbose:
            print(f"Cell with ID {cell_id} not found.")
        return

    fig = plt.figure(figsize=(12, 12))

    profile_mesh = cellobj.profile_mesh
    # Calculate the center of the contour
    center_x = np.mean(cellobj.contour.T[1])
    center_y = np.mean(cellobj.contour.T[0])

    # Calculate the new cropping boundaries based on the fixed crop size
    half_crop_size = crop_size / 2
    xmin = center_x - half_crop_size
    xmax = center_x + half_crop_size
    ymin = center_y - half_crop_size
    ymax = center_y + half_crop_size

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.imshow(image.image, cmap="gist_gray")
    plt.plot(cellobj.contour.T[1], cellobj.contour.T[0], "-", c="w", lw=3)

    plt.plot(
        [cellobj.y1[::2], cellobj.y2[::2]],
        [cellobj.x1[::2], cellobj.x2[::2]],
        c="cyan",
        lw=3,
    )
    plt.plot(cellobj.midline.T[1], cellobj.midline.T[0], "-", c="orange", lw=3)
    # Add scale bar
    scale_length_um = 1
    scale_length_px = scale_length_um / 0.065  # Convert µm to pixels
    scale_x = (xmin + xmax) / 2 - scale_length_px / 2  # Center the scale bar
    scale_y = ymin + (ymax - ymin) * 0.02  # Adjust the position of the scale bar
    plt.plot(
        [scale_x, scale_x + scale_length_px], [scale_y, scale_y], color="white", lw=5
    )
    # plt.scatter(profile_mesh[1][::2].T, profile_mesh[0][::2].T, s = 3, c = 'cyan')
    plt.title(
        f"Cell number: {cellobj.cell_id} with contour and profiling mesh", fontsize=16
    )
    plt.savefig(
        "C:/Users/u0158103/Documents/PhD/Pictures/coccus_mesh_svg.svg", format="svg"
    )
    plt.show()


def plot_width(cell, image, crop_size=70):
    cellobj = image.cells[cell]
    other_cellobj = image.cells[cell + 44]

    fig1, axs1 = plt.subplots(figsize=(12, 12))
    profile_mesh = cellobj.profile_mesh

    center_x = np.mean(cellobj.contour.T[1])
    center_y = np.mean(cellobj.contour.T[0])

    # Calculate the new cropping boundaries based on the fixed crop size
    half_crop_size = crop_size / 2
    xmin = center_x - half_crop_size
    xmax = center_x + half_crop_size
    ymin = center_y - half_crop_size
    ymax = center_y + half_crop_size

    axs1.set_position([0.1, 0.1, 0.5, 0.8])
    axs1.imshow(image.image, cmap="gist_gray")
    axs1.plot(cellobj.contour.T[1], cellobj.contour.T[0], lw=3, c="orange")

    axs1.set_xlim(xmin, xmax)
    axs1.set_ylim(ymin, ymax)
    # Add scale bar
    scale_length_um = 1
    scale_length_px = scale_length_um / 0.065  # Convert µm to pixels
    scale_x = (xmin + xmax) / 2 - scale_length_px / 2  # Center the scale bar
    scale_y = ymin + (ymax - ymin) * 0.02  # Adjust the position of the scale bar
    axs1.plot(
        [scale_x, scale_x + scale_length_px], [scale_y, scale_y], color="white", lw=5
    )
    axs1.set_xticks([])
    axs1.set_yticks([])
    axs1.set_title("Cell " + str(cell), fontname="Arial", fontsize=14)

    fig2, axs2 = plt.subplots(figsize=(6, 4))

    # Plot the cell width for the first cell (red curve)
    axs2.plot(cellobj.profiling_data["cell_widthno"], c="orange")

    # Plot the cell width for the other cell (green curve)
    axs2.plot(other_cellobj.profiling_data["cell_widthno"], c="b")

    # Set x-axis ticks and labels
    newxticks = [
        np.round(
            cellobj.contour_features["cell_length"] * (tick / profile_mesh.shape[1]), 1
        )
        for tick in axs2.get_xticks()
    ]
    axs2.set_xticks(axs2.get_xticks())
    axs2.set_xticklabels(newxticks, fontname="Arial", fontsize=12)

    # Set y-axis ticks and label
    newyticks = [np.round(tick, 1) for tick in axs2.get_yticks()]
    axs2.set_yticks(axs2.get_yticks())
    axs2.set_yticklabels(newyticks, fontname="Arial", fontsize=12)

    # Set x-axis to start at 0
    axs2.set_xlim(left=0)
    axs2.set_ylim(bottom=0)
    # Set x and y axis labels
    axs2.set_xlabel("cell length [µm]", fontname="Arial", fontsize=12)
    axs2.set_ylabel("cell width [µm]", fontname="Arial", fontsize=12)

    axs2.legend(["overlapping cell", "single cell"], frameon=False)

    fig3, axs3 = plt.subplots(figsize=(12, 12))
    other_profile_mesh = other_cellobj.profile_mesh

    center_x = np.mean(other_cellobj.contour.T[1])
    center_y = np.mean(other_cellobj.contour.T[0])

    # Calculate the new cropping boundaries based on the fixed crop size
    half_crop_size = crop_size / 2
    xmin = center_x - half_crop_size
    xmax = center_x + half_crop_size
    ymin = center_y - half_crop_size
    ymax = center_y + half_crop_size

    axs3.set_position([0.1, 0.1, 0.5, 0.8])
    axs3.imshow(image.image, cmap="gist_gray")
    axs3.plot(other_cellobj.contour.T[1], other_cellobj.contour.T[0], lw=3, c="b")

    axs3.set_xlim(xmin, xmax)
    axs3.set_ylim(ymin, ymax)

    axs3.set_xticks([])
    axs3.set_yticks([])
    axs3.set_title("Other Cell", fontname="Arial", fontsize=14)
    scale_length_um = 1
    scale_length_px = scale_length_um / 0.065  # Convert µm to pixels
    scale_x = (xmin + xmax) / 2 - scale_length_px / 2  # Center the scale bar
    scale_y = ymin + (ymax - ymin) * 0.02  # Adjust the position of the scale bar
    axs3.plot(
        [scale_x, scale_x + scale_length_px], [scale_y, scale_y], color="white", lw=5
    )

    fig1.savefig("C:/Users/u0158103/Documents/PhD/Pictures/bad_cell_image.svg", dpi=300)
    fig2.savefig("C:/Users/u0158103/Documents/PhD/Pictures/width.svg", dpi=300)
    fig3.savefig(
        "C:/Users/u0158103/Documents/PhD/Pictures/good_cell_image.svg", dpi=300
    )
    plt.show()


def plot_signal_profile(cell, image):
    cellobj = image.cells[cell]
    profile_mesh = cellobj.profile_mesh
    profiling_mesh = cellobj.profiling_data["phaco_mesh_intensity"]
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    xmin = np.min(profile_mesh[1] - 10)
    xmax = np.max(profile_mesh[1] + 10)
    ymin = np.min(profile_mesh[0] - 10)
    ymax = np.max(profile_mesh[0] + 10)
    ax[0].imshow(profiling_mesh, aspect="auto")
    ax[0].get_xticks()
    xticks = ax[0].get_xticks()
    newxticks = [
        np.round(
            cellobj.contour_features["cell_length"] * (tick / profiling_mesh.shape[1]),
            1,
        )
        for tick in xticks
    ]
    ax[0].set_xticklabels(newxticks, fontname="Arial", fontsize=12)
    ax[0].set_yticks([])
    ax[0].set_ylabel("signal\nstraighten image\n", fontname="Arial", fontsize=12)
    ax[0].set_xlabel("cell length [µm]", fontname="Arial", fontsize=12)

    ax[1].imshow(image, cmap="gist_gray", aspect="auto")
    ax[1].plot(cellobj.contour.T[1], cellobj.contour.T[0], "-", c="r")
    ax[1].plot(cellobj.midline.T[1], cellobj.midline.T[0], "-", c="y")
    ax[1].set_xlim(xmin, xmax)
    ax[1].set_ylim(ymin, ymax)
    plt.show()

from scipy.interpolate import interp1d

def plot_demograph_rotated(cell_lengths, normalized_average_mesh_intensity, title='Demograph Plot',
                           y_label='Normalized Distance From Midcell (µm)', x_label='Cell Length Percentile',
                           cmap='rainbow', cbar_title='DnaN-msfGFP Normalized Intensity'):
    # Filter out NaN values from cell_lengths and corresponding intensities
    valid_indices = [i for i, arr in enumerate(normalized_average_mesh_intensity) if arr is not None and not np.isnan(np.nanmean(arr))]
    cell_lengths = cell_lengths[valid_indices].reset_index(drop=True)
    normalized_average_mesh_intensity = normalized_average_mesh_intensity[valid_indices].reset_index(drop=True)
    
    
    # Sort the arrays based on cell_lengths
    sorted_indices = np.argsort(cell_lengths)
    sorted_lengths = cell_lengths[sorted_indices]
    sorted_intensities = normalized_average_mesh_intensity[sorted_indices]

    # Reset the indices to ensure correct alignment
    sorted_lengths = np.array(sorted_lengths)
    sorted_intensities = np.array(sorted_intensities)

    # Find the maximum length of the arrays
    max_length = max(len(arr) for arr in sorted_intensities)

    # Interpolate missing values for shorter arrays
    interpolated_arrays = []
    for array in sorted_intensities:
        x = np.arange(len(array))  # x-coordinates for the existing intensity values
        f = interp1d(x, array, kind='linear', fill_value='extrapolate')  # Interpolation function
        interpolated_array = f(np.linspace(0, len(array), max_length))  # Interpolated array
        interpolated_arrays.append(interpolated_array)
    stacked_demograph = np.vstack(interpolated_arrays)

    fig1 = plt.figure(figsize=(10, 8))  # Adjust figure size as needed

    ax = plt.subplot(111)
    image = ax.imshow(stacked_demograph.T, aspect='auto', cmap=cmap)  # Transpose the array
    cbar = plt.colorbar(image)  # Use the image as the mappable object for the colorbar
    cbar.set_label(cbar_title, rotation=90, labelpad=20, fontsize=14)  # Set colorbar label
    cbar.ax.tick_params(labelsize=14)  # Adjust colorbar tick label size

    # Calculate y-axis values and middle index
    y_axis_values = np.linspace(-1, 1, max_length)  # Ensure values are between -1 and 1
    middle_index = len(y_axis_values) // 2

    # Set y-axis ticks and labels
    plt.yticks([-0.5, middle_index, len(y_axis_values)-0.5], [-1, 0, 1])
    plt.xticks([0, len(sorted_lengths) * 0.25, len(sorted_lengths) * 0.5, len(sorted_lengths) * 0.75, len(sorted_lengths)],
               ['0', '25', '50', '75', '100'])
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    
    plt.title(title, fontsize=18)  # Set the plot title

    plt.show()