# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:02:27 2023

@author: Bart Steemans. Govers Lab.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from skimage.color import label2rgb
from skimage.measure import label
from . import utilities as u
import tifffile
from skimage.segmentation import find_boundaries
from scipy.spatial import cKDTree
dpi = 300

import numpy as np
import matplotlib.pyplot as plt


def _tutorial_contrast(image):
    """Return a display range without changing the measured image."""
    low, high = np.nanpercentile(image, (1, 99.7))
    return (low, high if high > low else low + 1)


def _tutorial_crop(shape, center, window_px):
    half = window_px // 2
    y = int(np.clip(center[0] - half, 0, max(shape[0] - window_px, 0)))
    x = int(np.clip(center[1] - half, 0, max(shape[1] - window_px, 0)))
    return slice(y, min(y + window_px, shape[0])), slice(x, min(x + window_px, shape[1]))


def pick_representative_cell(image_objects, margin_px=20):
    """Pick an interior, isolated, median-sized cell with valid geometry.

    The rule is deterministic and uses geometry only, never fluorescence intensity.
    Returns the image object and cell ID so the same cell can be revisited.
    """
    candidates = []
    for image in image_objects:
        height, width = image.image.shape
        for cell in getattr(image, "cells", []):
            contour = np.asarray(cell.contour)
            mesh = np.asarray(cell.mesh)
            if (contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 6
                    or mesh.ndim != 2 or mesh.shape[1] != 4 or len(mesh) < 4
                    or not np.isfinite(contour).all() or not np.isfinite(mesh).all()):
                continue
            low = contour.min(axis=0)
            high = contour.max(axis=0)
            if (low[0] < margin_px or low[1] < margin_px
                    or high[0] > height - margin_px or high[1] > width - margin_px):
                continue
            box = high - low
            long_side, short_side = max(box), min(box)
            if short_side < 5 or long_side / short_side < 1.5:
                continue
            center = contour.mean(axis=0)
            center_distance = np.linalg.norm(center / [height, width] - 0.5)
            candidates.append((image, cell.cell_id, long_side, center_distance, center))
    if not candidates:
        raise ValueError("No interior cells with valid elongated contours and meshes")
    median_length = np.median([item[2] for item in candidates])
    typical = [item for item in candidates if abs(item[2] - median_length) <= 0.25 * median_length]
    neighbors = {}
    for image in image_objects:
        group = [item for item in candidates if item[0] is image]
        if not group:
            continue
        all_centers = [np.asarray(cell.contour).mean(axis=0)
                       for cell in image.cells
                       if np.asarray(cell.contour).ndim == 2
                       and np.asarray(cell.contour).shape[1] == 2
                       and len(cell.contour) > 0]
        if len(all_centers) < 2:
            for item in group:
                neighbors[(image.image_name, item[1])] = np.inf
            continue
        distances = cKDTree(all_centers).query([item[4] for item in group], k=2)[0]
        for item, distance in zip(group, distances):
            neighbors[(image.image_name, item[1])] = distance[1] / item[2]
    winner = min(
        typical,
        key=lambda item: (-neighbors[(item[0].image_name, item[1])],
                          item[3], abs(item[2] - median_length),
                          item[0].image_name, item[1]),
    )
    return winner[0], winner[1]


def select_example_cell(image_objects, field_id=None, cell_id=None):
    """Choose an existing field/cell, or a representative cell in the chosen field."""
    objects = list(image_objects)
    if field_id is not None:
        objects = [image for image in objects if image.image_name.rsplit("_", 1)[0] == field_id]
        if not objects:
            raise ValueError(f"Field {field_id!r} was not found")
    if cell_id is None:
        return pick_representative_cell(objects)
    if field_id is None:
        raise ValueError("Set field_id when selecting a cell_id")
    for image in objects:
        if any(cell.cell_id == cell_id for cell in image.cells):
            return image, cell_id
    raise ValueError(f"Cell {cell_id} was not found in field {field_id}")


def _tutorial_channel_color(label):
    label = label.casefold()
    if "nucleoid" in label or "dapi" in label:
        return "#1DBCD3"
    if "rna" in label or "gfp" in label:
        return "#2BAE66"
    if "fm" in label or "membrane" in label:
        return "#D85755"
    return "#197E89"


def _tutorial_channel_cmap(label):
    color = _tutorial_channel_color(label)
    return LinearSegmentedColormap.from_list(
        f"bactoscoop_{label.replace(' ', '_')}", ["#000000", color]
    )


def plot_field_channels(dataset_dir, field_id, channel_labels, crop_size=650):
    """Preview the same pixel window across the named TIFF channels."""
    dataset_dir = Path(dataset_dir)
    labels = dict(channel_labels)
    if not labels:
        raise ValueError("channel_labels is empty")
    first = tifffile.imread(dataset_dir / f"{field_id}_{next(iter(labels))}.tiff")
    crop = _tutorial_crop(first.shape, (first.shape[0] // 2, first.shape[1] // 2), crop_size)
    fig, axes = plt.subplots(1, len(labels), figsize=(3.8 * len(labels), 4))
    for ax, (channel, label) in zip(np.atleast_1d(axes), labels.items()):
        image = tifffile.imread(dataset_dir / f"{field_id}_{channel}.tiff")
        if image.shape != first.shape:
            raise ValueError(f"{channel} dimensions do not match the first channel")
        data = image[crop]
        low, high = _tutorial_contrast(data)
        cmap = "gray" if channel == next(iter(labels)) else _tutorial_channel_cmap(label)
        ax.imshow(data, cmap=cmap, vmin=low, vmax=high)
        ax.set_title(f"{channel}  {label.upper()}", loc="left", fontsize=10, fontweight="bold", color="#193348")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"CHANNELS  /  {field_id}", x=0.02, y=0.98, ha="left", fontsize=14, fontweight="bold", color="#193348")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_segmentation_review(image_objects, crop_size=500, mask_label=None, center=None):
    """Compare phase and mask outlines in a selectable cell-centered crop."""
    objects = list(image_objects)
    if not objects:
        raise ValueError("image_objects is empty")
    fig, axes = plt.subplots(len(objects), 2, figsize=(10, 4.5 * len(objects)), squeeze=False)
    for row, image in enumerate(objects):
        if image.mask is None:
            raise ValueError(f"No mask is loaded for {image.image_name}")
        labels = np.unique(image.mask)
        labels = labels[labels > 0]
        if mask_label is not None and mask_label not in labels:
            raise ValueError(f"Mask label {mask_label} was not found in {image.image_name}")
        chosen = mask_label
        if chosen is None and center is None and len(labels):
            midpoint = np.asarray(image.image.shape) / 2
            nonzero = np.argwhere(image.mask > 0)
            closest = np.argmin(np.sum((nonzero - midpoint) ** 2, axis=1))
            chosen = image.mask[tuple(nonzero[closest])]
        if chosen is not None:
            focus = np.argwhere(image.mask == chosen).mean(axis=0)
        else:
            focus = center if center is not None else np.asarray(image.image.shape) / 2
        crop = _tutorial_crop(image.image.shape, focus, crop_size)
        phase, mask = image.image[crop], image.mask[crop]
        low, high = _tutorial_contrast(phase)
        for ax in axes[row]:
            ax.imshow(phase, cmap="gray", vmin=low, vmax=high)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        outlines = np.ma.masked_where(~find_boundaries(mask, mode="outer"), np.ones(mask.shape))
        axes[row, 1].imshow(outlines, cmap="autumn", vmin=0, vmax=1, alpha=0.9, interpolation="nearest")
        if chosen is not None:
            selected_outline = find_boundaries(mask == chosen, mode="outer")
            axes[row, 1].imshow(np.ma.masked_where(~selected_outline, np.ones(mask.shape)),
                                cmap="autumn", vmin=0, vmax=1,
                                interpolation="nearest")
        field = image.image_name.rsplit("_", 1)[0]
        axes[row, 0].set_title(f"{field}  |  phase", loc="left", fontsize=11)
        axes[row, 1].set_title(f"Mask outlines  |  selected label {chosen}", loc="left", fontsize=11)
    fig.suptitle("SEGMENTATION REVIEW", x=0.04, y=0.98, ha="left", fontsize=15, fontweight="bold", color="#193348")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def plot_cell_gallery(image_object, cell_id, channel="C2", window_px=110,
                      channel_label=None, align_signal=False, show_objects=False):
    """Show phase and mesh, optionally with a registered signal crop and objects.

    The registered crop uses the same local crop/phase registration called by
    ``object_detection(align=True)``. Plot it after detection to compare its
    detected object contours with the phase-cell contour.
    """
    cell = next((item for item in image_object.cells if item.cell_id == cell_id), None)
    if cell is None:
        raise ValueError(f"Cell {cell_id} was not found")
    if channel is not None and (not image_object.channels or channel not in image_object.channels):
        raise ValueError(f"Channel {channel} is not loaded")
    contour = np.asarray(cell.contour)
    mesh = np.asarray(cell.mesh)
    midline = np.asarray(cell.midline)
    extent = int(np.ceil(np.max(np.ptp(contour, axis=0)))) + 24
    crop = _tutorial_crop(image_object.image.shape, contour.mean(axis=0), max(window_px, extent))
    phase = image_object.image[crop]
    label = channel_label or channel
    panels = [(phase, "gray", "01  PHASE"), (phase, "gray", "02  CONTOUR + MESH")]
    signal_origin = (crop[1].start, crop[0].start)
    if channel is not None:
        signal = image_object.channels[channel][crop]
        if align_signal:
            signal, _, _, row_offset, column_offset = u.crop_image(
                image_object.channels[channel], cell.contour,
                phase_img=image_object.image,
            )
            signal_origin = (column_offset, row_offset)
        heading = "ALIGNED " if align_signal else ""
        panels.append((signal, _tutorial_channel_cmap(label),
                       f"03  {heading}{channel} {label.upper()}"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.9))
    for ax, (data, cmap, title) in zip(axes, panels):
        low, high = _tutorial_contrast(data)
        ax.imshow(data, cmap=cmap, vmin=low, vmax=high)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color="#193348")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    x0, y0 = crop[1].start, crop[0].start
    axes[1].plot(contour[:, 1] - x0, contour[:, 0] - y0, color="#FFCC66", lw=1.6)
    if channel is not None:
        axes[2].plot(contour[:, 1] - signal_origin[0], contour[:, 0] - signal_origin[1], color="#FFCC66", lw=1.6)
        if show_objects:
            for detected in cell.object_meshdata.get(channel, {}).get("object_contour", []):
                detected = np.asarray(detected)
                if detected.ndim == 2 and detected.shape[1] == 2:
                    axes[2].plot(detected[:, 1] - signal_origin[0],
                                 detected[:, 0] - signal_origin[1],
                                 color="#4DE6F2", lw=1.2)
    if mesh.ndim == 2 and mesh.shape[1] == 4:
        axes[1].plot(mesh[:, 1] - x0, mesh[:, 0] - y0, color="#36D6D0", lw=0.7)
        axes[1].plot(mesh[:, 3] - x0, mesh[:, 2] - y0, color="#36D6D0", lw=0.7)
        for segment in mesh[::3]:
            axes[1].plot([segment[1] - x0, segment[3] - x0], [segment[0] - y0, segment[2] - y0], color="#36D6D0", lw=0.55)
    if midline.ndim == 2 and midline.shape[1] == 2:
        axes[1].plot(midline[:, 1] - x0, midline[:, 0] - y0, color="#F26463", lw=1.3)
    max_scale_px = phase.shape[1] * 0.45
    scale_um = next(
        (value for value in (5, 2, 1, 0.5, 0.2, 0.1)
         if value / image_object.px <= max_scale_px),
        0.1,
    )
    scale_px = max(1, round(scale_um / image_object.px))
    axes[0].plot([7, 7 + scale_px], [phase.shape[0] - 9] * 2, color="white", lw=3, solid_capstyle="butt")
    axes[0].text(7, phase.shape[0] - 14, f"{scale_um:g} µm", color="white", fontsize=8)
    fig.suptitle(f"CELL {cell_id}  /  {image_object.image_name}", x=0.02, y=0.98, ha="left", fontsize=13, fontweight="bold", color="#193348")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig

def plot_C2_alignment(image_object, cell_id, window_px=110,
                      channel="C2", channel_label=None,
                      show_objects=False):
    """
    Compare the raw C2 signal with the phase-aligned C2 signal.

    BEFORE: original fluorescence image, cropped around the cell.
    AFTER:  fluorescence crop returned by u.crop_image(..., phase_img=...).

    Cell contours are overlaid using the correct origin for each crop.
    Detected object contours can optionally be shown on the aligned panel.
    """

    cell = next(
        (item for item in image_object.cells if item.cell_id == cell_id),
        None,
    )
    if cell is None:
        raise ValueError(f"Cell {cell_id} was not found")

    if not image_object.channels or channel not in image_object.channels:
        raise ValueError(f"Channel {channel} is not loaded")

    contour = np.asarray(cell.contour)
    label = channel_label or channel

    # BEFORE: use the original image and the same cropping method
    # as plot_cell_gallery().
    extent = int(np.ceil(np.max(np.ptp(contour, axis=0)))) + 24

    crop = _tutorial_crop(
        image_object.image.shape,
        contour.mean(axis=0),
        max(window_px, extent),
    )

    signal_before = image_object.channels[channel][crop]

    # Crop origin in image coordinates: (column, row).
    origin_before = (
        crop[1].start,
        crop[0].start,
    )

    # AFTER: apply the same local registration as in plot_cell_gallery().
    signal_after, _, _, row_offset, column_offset = u.crop_image(
        image_object.channels[channel],
        cell.contour,
        phase_img=image_object.image,
    )

    # The function returns row offset first, column offset second.
    # For plotting, x = column and y = row.
    origin_after = (
        column_offset,
        row_offset,
    )

    panels = [
        (signal_before, origin_before, f"01  UNALIGNED {channel}"),
        (signal_after, origin_after, f"02  ALIGNED {channel}"),
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4.8),
    )

    # Use identical contrast limits in both panels.
    combined = np.concatenate([
        signal_before.ravel(),
        signal_after.ravel(),
    ])
    combined = combined[np.isfinite(combined)]

    if combined.size == 0:
        raise ValueError("Both signal crops contain no finite values")

    low, high = np.percentile(combined, [1, 99.5])

    if low == high:
        high = low + 1

    for ax, (signal, origin, title) in zip(axes, panels):

        ax.imshow(
            signal,
            cmap=_tutorial_channel_cmap(label),
            vmin=low,
            vmax=high,
        )

        ax.set_title(
            title,
            loc="left",
            fontsize=11,
            fontweight="bold",
            color="#193348",
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Cell contours use (row, column).
        # Matplotlib uses (x=column, y=row).
        x0, y0 = origin

        ax.plot(
            contour[:, 1] - x0,
            contour[:, 0] - y0,
            color="#FFCC66",
            lw=1.6,
        )

    # Show detected objects only on the aligned panel.
    # These contours are taken directly from the cell object.
    if show_objects:
        x0, y0 = origin_after

        for detected in cell.object_meshdata.get(
            channel, {}
        ).get("object_contour", []):

            detected = np.asarray(detected)

            if detected.ndim == 2 and detected.shape[1] == 2:
                axes[1].plot(
                    detected[:, 1] - x0,
                    detected[:, 0] - y0,
                    color="#4DE6F2",
                    lw=1.2,
                )

    fig.suptitle(
        f"{channel} ALIGNMENT  /  CELL {cell_id}  /  "
        f"{image_object.image_name}",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#193348",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))

    return fig
def snapshot_cell_contours(image_objects):
    """Copy cell IDs and contours before curation changes the collection."""
    return {
        image.image_name: {
            cell.cell_id: np.asarray(cell.contour).copy() for cell in image.cells
        }
        for image in image_objects
    }


def plot_curation_review(before_contours, image_objects, crop_size=700):
    """Show retained cells in green and removed cells in red on phase images.

    ``before_contours`` is made by :func:`snapshot_cell_contours` before
    curation. Cells are matched by image name and cell ID.
    """
    objects = list(image_objects)
    if not objects:
        raise ValueError("image_objects is empty")
    fig, axes = plt.subplots(1, len(objects), figsize=(5.2 * len(objects), 5.2), squeeze=False)
    for ax, image in zip(axes[0], objects):
        original = before_contours.get(image.image_name)
        if original is None:
            raise ValueError(f"No pre-curation contours for {image.image_name}")
        kept = {cell.cell_id for cell in image.cells}
        if not kept.issubset(original):
            raise ValueError(f"Curated cell IDs do not match the original meshes for {image.image_name}")
        for cell in image.cells:
            old_center = np.asarray(original[cell.cell_id]).mean(axis=0)
            new_center = np.asarray(cell.contour).mean(axis=0)
            if np.linalg.norm(old_center - new_center) > 1:
                raise ValueError(f"Curated cell geometry does not match the original mesh for {image.image_name}")
        crop = _tutorial_crop(
            image.image.shape,
            (image.image.shape[0] // 2, image.image.shape[1] // 2),
            crop_size,
        )
        phase = image.image[crop]
        low, high = _tutorial_contrast(phase)
        ax.imshow(phase, cmap="gray", vmin=low, vmax=high)
        x0, y0 = crop[1].start, crop[0].start
        for cell_id, contour in original.items():
            contour = np.asarray(contour)
            margin = 3
            fully_inside = (
                (contour[:, 0] >= crop[0].start + margin)
                & (contour[:, 0] < crop[0].stop - margin)
                & (contour[:, 1] >= crop[1].start + margin)
                & (contour[:, 1] < crop[1].stop - margin)
            )
            if not fully_inside.all():
                continue
            keep = cell_id in kept
            ax.plot(
                contour[:, 1] - x0, contour[:, 0] - y0,
                color="#34D986" if keep else "#FF686B",
                lw=1.15 if keep else 0.85,
                alpha=0.95 if keep else 0.82,
            )
        field = image.image_name.rsplit("_", 1)[0]
        ax.set_title(
            f"{field}\n{len(kept):,} kept  ·  {len(original) - len(kept):,} removed",
            loc="left", fontsize=11, color="#193348",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.legend(
        handles=[
            Line2D([], [], color="#34D986", lw=2, label="Kept"),
            Line2D([], [], color="#FF686B", lw=2, label="Removed"),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("CURATION REVIEW", x=0.02, y=1.04, ha="left", fontsize=15, fontweight="bold", color="#193348")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


def plot_signal_gallery(image_object, cell_id, channel_labels=None,
                        window_px=120, show_objects=False):
    """Show the phase image beside raw channel crops, with optional detections.

    Cell contours are displayed on all panels. Detected object contours
    are optionally overlaid on the corresponding fluorescence channels.

    Crops share pixel coordinates. This function does not register channels.
    """
    labels = dict(channel_labels or {"C4": "RNA", "C5": "DAPI / nucleoid"})
    if not labels:
        raise ValueError("channel_labels is empty")

    cell = next(
        (item for item in image_object.cells if item.cell_id == cell_id),
        None,
    )
    if cell is None:
        raise ValueError(f"Cell {cell_id} was not found")

    missing = [
        channel for channel in labels
        if not image_object.channels or channel not in image_object.channels
    ]
    if missing:
        raise ValueError(f"Channels not loaded: {missing}")

    contour = np.asarray(cell.contour)
    crop = _tutorial_crop(
        image_object.image.shape, contour.mean(axis=0), window_px
    )
    x0, y0 = crop[1].start, crop[0].start

    # Prepare panels
    panels = [("C1  PHASE", image_object.image, "gray", None)]

    for channel, label in labels.items():
        cmap = _tutorial_channel_cmap(label)
        panels.append((
            f"{channel}  {label.upper()}",
            image_object.channels[channel],
            cmap,
            channel,
        ))

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(3.7 * len(panels), 3.9),
    )

    for ax, (title, full_image, cmap, channel) in zip(axes, panels):
        data = full_image[crop]
        low, high = _tutorial_contrast(data)

        ax.imshow(data, cmap=cmap, vmin=low, vmax=high)
        ax.set_title(
            title, loc="left", fontsize=10,
            fontweight="bold", color="#193348",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Plot cell contour on every panel
        ax.plot(
            contour[:, 1] - x0,
            contour[:, 0] - y0,
            color="#FFCF70",
            lw=1.5,
        )

        # Optionally plot detected objects on fluorescence panels
        if channel is not None and show_objects:
            contours = cell.object_meshdata.get(
                channel, {}
            ).get("object_contour", [])

            if contours is not None:
                for item in contours:
                    outline = np.asarray(item)

                    if outline.ndim == 2 and outline.shape[1] == 2:
                        ax.plot(
                            outline[:, 1] - x0,
                            outline[:, 0] - y0,
                            color="#4DE6F2",
                            lw=1.2,
                        )

    fig.suptitle(
        f"CHANNEL EXAMPLE  /  CELL {cell_id}",
        x=0.02, y=0.98, ha="left",
        fontsize=14, fontweight="bold", color="#193348",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_object_area_vs_cell_area(features, channel_labels):
    """Compare detected object area with cell area for each named channel."""
    labels = dict(channel_labels)
    if not labels:
        raise ValueError("channel_labels is empty")
    if "cell_area" not in features:
        raise ValueError("Missing cell_area")
    fig, axes = plt.subplots(1, len(labels), figsize=(4.3 * len(labels), 4), squeeze=False)
    x = np.asarray(features["cell_area"], dtype=float)
    for ax, (channel, label) in zip(axes[0], labels.items()):
        column = f"{channel}_cell_total_obj_area"
        if column not in features:
            raise ValueError(f"Missing {column}; calculate object features first")
        y = np.asarray(features[column], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=12, color=_tutorial_channel_color(label), alpha=0.42, linewidths=0)
        ax.set_title(f"{channel}  {label}", loc="left", fontsize=11, fontweight="bold", color="#193348")
        ax.set_xlabel("Cell area (µm²)")
        ax.set_ylabel("Detected object area (µm²)")
        ax.text(0.97, 0.96, f"n = {valid.sum():,}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#516676")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#E7EDF1", lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#516676", labelsize=9)
    fig.suptitle("SUBCELLULAR AREA  /  CELL AREA", x=0.02, y=1.04, ha="left", fontsize=14, fontweight="bold", color="#193348")
    fig.tight_layout()
    return fig


def plot_feature_overview(features, signal_column="C4_NC_ratio"):
    """
    Exploratory overview of cell morphology, constriction,
    detected objects, and relative signal localization.

    C2 = outer membrane
    C3 = inner membrane
    C4 = RNA
    C5 = DAPI

    Missing features are indicated in their panels.
    Nothing is saved.
    """

    def numeric(column):
        if column not in features.columns:
            return None
        return pd.to_numeric(features[column], errors="coerce").to_numpy()

    def unavailable(ax, *columns):
        ax.text(
            0.5, 0.5,
            "Feature unavailable:\n" + "\n".join(columns),
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=9, color="gray",
        )
        ax.set_axis_off()

    length = numeric("cell_length")
    width = numeric("cell_width")
    constriction = numeric("cell_constriction_degree_width_axial")

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 9),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E7EDF1", lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    # 1. Cell morphology, coloured by constriction
    ax = axes[0]

    if length is not None and width is not None:
        valid = np.isfinite(length) & np.isfinite(width)

        if constriction is not None:
            valid &= np.isfinite(constriction)

            scatter = ax.scatter(
                length[valid],
                width[valid],
                c=constriction[valid],
                cmap="viridis",
                s=10,
                alpha=0.55,
                linewidths=0,
            )

            fig.colorbar(
                scatter,
                ax=ax,
                label="Constriction degree",
            )
        else:
            ax.scatter(
                length[valid],
                width[valid],
                color="#197E89",
                s=10,
                alpha=0.4,
                linewidths=0,
            )

        ax.set(
            xlabel="Cell length (µm)",
            ylabel="Cell width (µm)",
            title="Cell shape and constriction",
        )
    else:
        unavailable(ax, "cell_length", "cell_width")

    # 2. Constriction-degree distribution
    ax = axes[1]

    if constriction is not None:
        valid = constriction[np.isfinite(constriction)]

        ax.hist(
            valid,
            bins=35,
            color="#197E89",
            edgecolor="white",
        )

        ax.set(
            xlabel="Constriction degree",
            ylabel="Cells",
            title="Cell constriction",
        )
    else:
        unavailable(ax, "cell_constriction_degree_width_axial")

    # 3. Number of detected DAPI objects per cell
    ax = axes[2]
    object_number = numeric("C5_object_number")

    if object_number is not None:
        counts = (
            pd.Series(object_number)
            .dropna()
            .value_counts()
            .sort_index()
        )

        ax.bar(
            counts.index,
            counts.values,
            color="#7F3F98",
            width=0.75,
        )

        ax.set(
            xlabel="Detected DAPI objects per cell",
            ylabel="Cells",
            title="DAPI object distribution",
        )
    else:
        unavailable(ax, "C5_object_number")

    # 4. Nucleoid-to-cell ratio
    ax = axes[3]
    nc_ratio = numeric("C5_NC_ratio")

    if nc_ratio is not None:
        valid = nc_ratio[np.isfinite(nc_ratio)]

        ax.hist(
            valid,
            bins=35,
            color="#7F3F98",
            edgecolor="white",
        )

        ax.set(
            xlabel="DAPI NC ratio",
            ylabel="Cells",
            title="Nucleoid occupancy",
        )
    else:
        unavailable(ax, "C5_NC_ratio")

    # 5. Relative object occupancy versus cell length
    ax = axes[4]
    signal = numeric(signal_column)

    if length is not None and signal is not None:
        valid = np.isfinite(length) & np.isfinite(signal)

        ax.scatter(
            length[valid],
            signal[valid],
            color="#E17953",
            s=10,
            alpha=0.4,
            linewidths=0,
        )

        ax.set(
            xlabel="Cell length (µm)",
            ylabel=signal_column.replace("_", " "),
            title="Object occupancy versus cell length",
        )
    else:
        unavailable(ax, "cell_length", signal_column)

    # 6. Outer- versus inner-membrane signal constriction
    ax = axes[5]

    outer = numeric("C2_signal_constriction_degree_mesh")
    inner = numeric("C3_signal_constriction_degree_mesh")

    if outer is not None and inner is not None:
        valid = np.isfinite(outer) & np.isfinite(inner)

        ax.scatter(
            outer[valid],
            inner[valid],
            color="#36A5A0",
            s=10,
            alpha=0.4,
            linewidths=0,
        )

        ax.set(
            xlabel="Outer membrane constriction degree (C2)",
            ylabel="Inner membrane constriction degree (C3)",
            title="Membrane signal localization",
        )
    else:
        unavailable(
            ax,
            "C2_signal_constriction_degree_mesh",
            "C3_signal_constriction_degree_mesh",
        )

    fig.suptitle(
        f"CELL FEATURE OVERVIEW  /  {len(features):,} cells",
        fontsize=15,
        fontweight="bold",
        color="#193348",
    )

    return fig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_histogram(
    df,
    feature,
    bins=40,
    xlabel=None,
    color="#197E89",
    figsize=(7, 4.5),
):
    """Plot a feature distribution with mean, median and interquartile range."""

    if feature not in df.columns:
        raise ValueError(f"Feature not found: {feature}")

    values = pd.to_numeric(df[feature], errors="coerce").to_numpy()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(f"No valid values for {feature}")

    mean = np.mean(values)
    median = np.median(values)
    q1, q3 = np.percentile(values, [25, 75])

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    ax.hist(
        values,
        bins=bins,
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )

    # Interquartile range
    ax.axvspan(
        q1, q3,
        color=color,
        alpha=0.12,
        label=f"IQR: {q1:.2f}–{q3:.2f}",
    )

    # Mean and median
    ax.axvline(
        mean,
        color="#E17953",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean: {mean:.2f}",
    )

    ax.axvline(
        median,
        color="#193348",
        linestyle="-",
        linewidth=1.8,
        label=f"Median: {median:.2f}",
    )

    ax.set_xlabel(xlabel or feature.replace("_", " "), fontsize=11)
    ax.set_ylabel("Number of cells", fontsize=11)
    ax.set_title(
        f"{feature.replace('_', ' ')}\nN = {len(values):,} cells",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )

    ax.legend(frameon=False, fontsize=9)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def plot_feature_scatter(
    df,
    x_feature,
    y_feature,
    color_by=None,
    xlabel=None,
    ylabel=None,
    trendline=True,
    figsize=(7, 5),
):
    """
    Plot two features against each other.

    Optionally colour points by a third numeric feature.
    A linear regression line and Pearson correlation can be shown.
    """

    columns = [x_feature, y_feature]

    if color_by is not None:
        columns.append(color_by)

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Features not found: {missing}")

    # Convert selected features to numeric values
    data = df[columns].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid cells remain for the selected features")

    x = data[x_feature].to_numpy()
    y = data[y_feature].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    if color_by is None:
        ax.scatter(
            x, y,
            s=13,
            alpha=0.3,
            color="#197E89",
            edgecolors="none",
            rasterized=True,
        )
    else:
        scatter = ax.scatter(
            x, y,
            c=data[color_by],
            cmap="viridis",
            s=13,
            alpha=0.5,
            edgecolors="none",
            rasterized=True,
        )

        fig.colorbar(
            scatter,
            ax=ax,
            label=color_by.replace("_", " "),
        )

    # Linear trend and Pearson correlation
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:

        r = np.corrcoef(x, y)[0, 1]

        if trendline:
            slope, intercept = np.polyfit(x, y, 1)

            x_line = np.linspace(x.min(), x.max(), 200)

            ax.plot(
                x_line,
                slope * x_line + intercept,
                color="#E17953",
                linewidth=2,
                linestyle="--",
                label=f"Linear fit (r = {r:.2f})",
            )

            ax.legend(frameon=False)

        else:
            ax.text(
                0.03, 0.97,
                f"Pearson r = {r:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

    ax.set_xlabel(xlabel or x_feature.replace("_", " "), fontsize=11)
    ax.set_ylabel(ylabel or y_feature.replace("_", " "), fontsize=11)

    ax.set_title(
        f"{x_feature.replace('_', ' ')} vs. {y_feature.replace('_', ' ')}"
        f"\nN = {len(data):,} cells",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig

def plot_contour(
    cell_id,
    image,
    base_channel=None,
    verbose=False,
    add_scalebar=True,
    scalebar_um=1,
    um_per_px=0.065,
    scalebar_color="white",
    scalebar_lw=6,
    window_px=None,        # <-- NEW: fixed window size (square)
    save_svg_path=None,
):
    """
    Plot the contour of a cell with optional scalebar.

    Parameters
    ----------
    window_px : int or None
        If provided, creates a fixed square window (in pixels) centered on the cell.
        If None, tight crop is used (default behavior).
    """

    cellobj = next((cell for cell in image.cells if cell.cell_id == cell_id), None)

    if cellobj is None:
        if verbose:
            print(f"Cell with ID {cell_id} not found.")
        return

    contour = cellobj.contour
    full_img = image.image

    # ---------------------------
    # Fixed window crop
    # ---------------------------
    if window_px is not None:

        # center of cell
        cy = int(np.mean(contour[:, 0]))
        cx = int(np.mean(contour[:, 1]))

        half = window_px // 2

        y_min = max(cy - half, 0)
        y_max = min(cy + half, full_img.shape[0])
        x_min = max(cx - half, 0)
        x_max = min(cx + half, full_img.shape[1])

        cropped_img = full_img[y_min:y_max, x_min:x_max]

        adjusted_contour_x = contour[:, 1] - x_min
        adjusted_contour_y = contour[:, 0] - y_min

    # ---------------------------
    # Default tight crop
    # ---------------------------
    else:
        cropped_img, _, _, x_offset, y_offset = u.crop_image(
            image=full_img, contour=contour, mask_to_crop=None, phase_img=None
        )

        adjusted_contour_x = contour[:, 1] - y_offset
        adjusted_contour_y = contour[:, 0] - x_offset

    # ---------------------------
    # SVG settings (editable)
    # ---------------------------
    plt.rcParams.update({
        "font.family": "Arial",
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cropped_img, cmap="gist_gray")

    ax.plot(adjusted_contour_x, adjusted_contour_y, "-", c="w", lw=3)

    # ---------------------------
    # Scalebar (vector)
    # ---------------------------
    if add_scalebar:
        bar_px = scalebar_um / um_per_px
        h, w = cropped_img.shape[:2]

        x_start = int(w * 0.05)
        y_start = int(h * 0.95)

        ax.plot(
            [x_start, x_start + bar_px],
            [y_start, y_start],
            color=scalebar_color,
            lw=scalebar_lw,
            solid_capstyle="butt",
        )

        ax.text(
            x_start + bar_px / 2,
            y_start - h * 0.05,
            f"{scalebar_um:.1f} µm",
            color=scalebar_color,
            ha="center",
            va="bottom",
            fontsize=12,
            fontfamily="Arial",
        )

    ax.set_title(
        f"Cell {cellobj.cell_id} Frame {image.frame}",
        fontfamily="Arial"
    )

    ax.axis("off")
    fig.tight_layout(pad=0)

    if save_svg_path:
        fig.savefig(
            save_svg_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0
        )

    plt.show()
    plt.close(fig)



    # plt.plot(midline.T[1], midline.T[0], '-', c='blue', lw=5)
    # plt.plot([msh[:,1][::2], msh[:,3][::2]], [msh[:,0][::2], msh[:,2][::2]], '-', c='cyan', lw=2)
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_cell_contour_gallery(
    cell_id,
    image,
    channels,
    window_px=100,
    phase_first=True,
    phase_cmap="gist_grey",
    channel_cmap="gist_grey",
    contour_color="yellow",
    contour_lw=1.5,
    add_scalebar=True,
    scalebar_um=1,
    um_per_px=0.065,
    scalebar_color="white",
    scalebar_lw=4,
    title=True,
    figsize_per_panel=3.0,
    save_path=None,
    save_individual_dir=None,
    dpi=500,
    verbose=False,
):
    """
    Plot a cropped gallery for one cell using:
      - image.image for phase contrast
      - image.channels[ch] for channel images
      - only the cell contour overlay

    Parameters
    ----------
    cell_id : int or str
    image : object
        Typically ic.image_objects[frame_id]
    channels : list of str
        Example: ["C1", "C2", "C3", "C4", "C5"]
    """

    cellobj = next((cell for cell in image.cells if cell.cell_id == cell_id), None)
    if cellobj is None:
        if verbose:
            print(f"Cell with ID {cell_id} not found.")
        return None

    contour = np.asarray(cellobj.contour)   # expected shape: [N, 2] with [y, x]
    full_phase = image.image                # phase contrast image

    if full_phase is None:
        if verbose:
            print("Phase image not found at image.image")
        return None

    # fixed square crop centered on the cell
    cy = int(np.mean(contour[:, 0]))
    cx = int(np.mean(contour[:, 1]))
    half = int(window_px // 2)

    y_min = max(cy - half, 0)
    y_max = min(cy + half, full_phase.shape[0])
    x_min = max(cx - half, 0)
    x_max = min(cx + half, full_phase.shape[1])

    adjusted_contour_x = contour[:, 1] - x_min
    adjusted_contour_y = contour[:, 0] - y_min

    panels = []

    if phase_first:
        panels.append(("Phase", full_phase[y_min:y_max, x_min:x_max], phase_cmap))

    for ch in channels:
        if ch not in image.channels:
            if verbose:
                print(f"Channel {ch} not found in image.channels, skipping.")
            continue
        ch_img = image.channels[ch]
        cropped = ch_img[y_min:y_max, x_min:x_max]
        panels.append((str(ch), cropped, channel_cmap))

    if len(panels) == 0:
        if verbose:
            print("No panels available to plot.")
        return None

    plt.rcParams.update({
        "font.family": "Arial",
        "svg.fonttype": "none",
    })

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(figsize_per_panel * n_panels, figsize_per_panel),
        squeeze=False
    )
    axes = axes.ravel()

    def _add_scalebar(ax, img_shape):
        if not add_scalebar:
            return
        h, w = img_shape[:2]
        bar_px = scalebar_um / um_per_px
        x_start = int(w * 0.05)
        y_start = int(h * 0.93)

        ax.plot(
            [x_start, x_start + bar_px],
            [y_start, y_start],
            color=scalebar_color,
            lw=scalebar_lw,
            solid_capstyle="butt",
        )
        ax.text(
            x_start + bar_px / 2,
            y_start - h * 0.04,
            f"{scalebar_um:.1f} µm",
            color=scalebar_color,
            ha="center",
            va="bottom",
            fontsize=10,
            fontfamily="Arial",
        )

    for ax, (panel_name, cropped_img, cmap) in zip(axes, panels):
        ax.imshow(cropped_img, cmap=cmap)
        ax.plot(adjusted_contour_x, adjusted_contour_y, color=contour_color, lw=contour_lw)
        _add_scalebar(ax, cropped_img.shape)

        if title:
            ax.set_title(panel_name)

        ax.axis("off")

    fig.tight_layout(pad=0.05)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)

    if save_individual_dir is not None:
        os.makedirs(save_individual_dir, exist_ok=True)

        for panel_name, cropped_img, cmap in panels:
            fig_i, ax_i = plt.subplots(figsize=(figsize_per_panel, figsize_per_panel))
            ax_i.imshow(cropped_img, cmap=cmap)
            ax_i.plot(adjusted_contour_x, adjusted_contour_y, color=contour_color, lw=contour_lw)
            _add_scalebar(ax_i, cropped_img.shape)

            if title:
                ax_i.set_title(panel_name)

            ax_i.axis("off")
            fig_i.tight_layout(pad=0.05)

            safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(panel_name))
            out_path = os.path.join(save_individual_dir, f"cell_{cell_id}_{safe_name}.png")
            fig_i.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
            plt.close(fig_i)

    plt.show()

    return {
        "cell_id": cell_id,
        "crop_bounds": (y_min, y_max, x_min, x_max),
        "panel_names": [p[0] for p in panels],
        "figure": fig,
    }

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
        A list of (cell_id, frame) pairs for cells to be plotted.
    image_objects : list
        A list of image objects corresponding to the frames in frame_cell_pairs.
    message : str, optional
        An optional message to be added to the title of the plot.
    """
    images_by_frame = {}
    for image_obj in image_objects:
        if image_obj.frame in images_by_frame:
            raise ValueError(f"Duplicate image frame {image_obj.frame}.")
        images_by_frame[image_obj.frame] = image_obj

    for cell_id, frame in frame_cell_pairs:
        image_obj = images_by_frame.get(frame)
        if image_obj is None:
            raise ValueError(f"No image object has frame {frame}.")
        cellobj = next(
            (cell for cell in image_obj.cells if cell.cell_id == cell_id), None
        )
        if cellobj is None:
            raise ValueError(f"Cell {cell_id} is absent from frame {frame}.")
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
    feature_dataframe, channels, selected_frame=None, selected_cell_id=None,
    show=True,
):
    """Plot normalized axial profiles for selected cells."""
    profile_columns = [f"{channel}_axial_intensity" for channel in channels]
    missing = set(profile_columns) - set(feature_dataframe.columns)
    if missing:
        raise ValueError(f"Missing axial profile columns: {sorted(missing)}")
    mask = np.ones(len(feature_dataframe), dtype=bool)
    if selected_frame is not None:
        mask &= feature_dataframe["frame"].to_numpy() == selected_frame
    if selected_cell_id is not None:
        mask &= feature_dataframe["cell_id"].to_numpy() == selected_cell_id
    selected = feature_dataframe.loc[mask, ["frame", "cell_id", *profile_columns]]
    figures = []
    colors = {"C2": "#E3484D", "C3": "#CF42BA", "C4": "#27AA59", "C5": "#3186DE"}
    for _, row in selected.drop_duplicates(["frame", "cell_id"]).iterrows():
        fig, ax = plt.subplots(figsize=(8, 4))
        for channel, column in zip(channels, profile_columns):
            values = np.asarray(row[column], dtype=float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            low, high = finite.min(), finite.max()
            scaled = (values - low) / (high - low) if high > low else np.zeros_like(values)
            ax.plot(np.linspace(0, 1, len(scaled)), scaled, label=channel,
                    color=colors.get(channel), lw=1.8)
        ax.set(xlabel="Relative position along cell", ylabel="Normalized intensity",
               ylim=(-0.03, 1.08), title=f"Frame {row['frame']}  |  Cell {row['cell_id']}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#E7EDF1", lw=0.8)
        ax.legend(frameon=False, ncol=len(channels))
        fig.tight_layout()
        figures.append(fig)
        if show:
            plt.show()
    return figures


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


import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def plot_object_contours(
    cell_id,
    image,
    contour_channels,
    shift=False,
    base_channel=None,
    verbose=False,
    add_scalebar=True,
    scalebar_um=1,
    um_per_px=0.065,
    scalebar_color="white",
    scalebar_lw=8,
    window_px=None,          # fixed square window size (pixels); if None -> original crop_image behavior
    save_svg_path=None,  
    save_tiff_path=None    # SVG only (editable in Illustrator)
):
    """
    Plot the contours of a cell and its objects for multiple channels on a single base channel image.

    Behaves like your original version:
      - base_channel None -> uses image.image
      - shift True -> crop_image(... phase_img=image.image)
      - else -> crop_image(... phase_img=None)
      - plots cell contour + object contours from cropped_object_contour
      - adds scalebar the same way

    Added:
      - window_px: optional fixed square window around cell center (bypasses crop_image).
      - save_svg_path: save editable SVG (scalebar/text/contours editable in Illustrator).
    """

    cellobj = next((cell for cell in image.cells if cell.cell_id == cell_id), None)
    if cellobj is None:
        if verbose:
            print(f"Cell with ID {cell_id} not found.")
        return

    contour = cellobj.contour

    # Keep SVG text editable
    plt.rcParams.update({
        "font.family": "Arial",
        "svg.fonttype": "none",
    })

    # ---------------------------------------------------------
    # 1) Get cropped_img and cropped_contour exactly like before
    #    unless window_px is requested.
    # ---------------------------------------------------------
    if window_px is None:
        # Original logic (respects base_channel and shift exactly)
        if base_channel is None:
            cropped_img, _, cropped_contour, x_offset, y_offset = u.crop_image(
                image=image.image,
                contour=contour,
                mask_to_crop=None,
                phase_img=None,
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

        # Object contours: use exactly what your original code used
        # (already in cropped coords)
        cropped_object_contours = {}
        for chann in contour_channels:
            # keep it robust if key missing
            cropped_object_contours[chann] = (
                cellobj.object_meshdata.get(chann, {}).get("cropped_object_contour", None)
            )

    else:
        # ---------------------------------------------------------
        # Fixed window mode (consistent output size across cells)
        # This mode does NOT call crop_image, so we must:
        #  - choose the same base image selection logic
        #  - shift the cell contour + object contours into window coords
        # ---------------------------------------------------------
        if base_channel is None:
            base_img = image.image
        else:
            base_img = image.channels[base_channel]

        # center of cell in full image coords
        cy = int(np.mean(contour[:, 0]))
        cx = int(np.mean(contour[:, 1]))
        half = int(window_px // 2)

        y_min = max(cy - half, 0)
        y_max = min(cy + half, base_img.shape[0])
        x_min = max(cx - half, 0)
        x_max = min(cx + half, base_img.shape[1])

        cropped_img = base_img[y_min:y_max, x_min:x_max]

        # Cell contour in window coords
        cropped_contour = np.column_stack([contour[:, 0] - y_min, contour[:, 1] - x_min])

        # Object contours:
        # In many pipelines, cropped_object_contour is in crop_image coords,
        # so for window mode we instead try to use the full-res object contour
        # if available, otherwise we fall back to cropped_object_contour (unchanged).
        cropped_object_contours = {}
        for chann in contour_channels:
            meshdata = cellobj.object_meshdata.get(chann, {})

            # Try common keys for full-image object contours
            full_key_candidates = ["object_contour", "object_contours", "raw_object_contour", "raw_object_contours"]
            full_contours = None
            for k in full_key_candidates:
                if k in meshdata and meshdata[k] is not None:
                    full_contours = meshdata[k]
                    break

            if full_contours is not None:
                # Shift full contours into window coords
                shifted = []
                for obj in full_contours:
                    obj = np.asarray(obj)
                    shifted.append(np.column_stack([obj[:, 0] - y_min, obj[:, 1] - x_min]))
                cropped_object_contours[chann] = shifted
            else:
                # Fallback: use cropped_object_contour as-is (may not align in window mode,
                # but prevents hard failure if only that exists)
                cropped_object_contours[chann] = meshdata.get("cropped_object_contour", None)

    # ---------------------------------------------------------
    # 2) Plot (same operations as your original version)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cropped_img, cmap="gist_grey")
    ax.plot(cropped_contour.T[1], cropped_contour.T[0], color="yellow")

    for chann in contour_channels:
        ccont = cropped_object_contours.get(chann, None)
        if ccont is not None:
            for obj_contour in ccont:
                ax.plot(
                    obj_contour.T[1],
                    obj_contour.T[0],
                    "-",
                    lw=2,
                    label=f"Channel {chann}",
                )

    if contour_channels:
        ax.legend(frameon=False)

    # ---------------------------------------------------------
    # 3) Scalebar (same logic; editable in SVG)
    # ---------------------------------------------------------
    if add_scalebar:
        bar_px = scalebar_um / um_per_px
        h, w = cropped_img.shape[:2]

        x_start = int(w * 0.05)
        y_start = int(h * 0.95)

        ax.plot(
            [x_start, x_start + bar_px],
            [y_start, y_start],
            color=scalebar_color,
            lw=scalebar_lw,
            solid_capstyle="butt",
        )
        ax.text(
            x_start + bar_px / 2,
            y_start - h * 0.03,
            f"{scalebar_um:.1f} µm",
            color=scalebar_color,
            ha="center",
            va="bottom",
            fontsize=20,
            fontfamily="Arial",
        )

    ax.set_title(f"Cell {cellobj.cell_id} - Frame {image.frame}— Base Channel {base_channel}", fontfamily="Arial")
    ax.axis("off")
    fig.tight_layout(pad=0)
    if save_tiff_path:
        plt.imsave(
            save_tiff_path,
            cropped_img,
            cmap="gist_grey",
            format="tiff"
        )
    # SVG only (editable)
    if save_svg_path:
        fig.savefig(save_svg_path, format="svg", bbox_inches="tight", pad_inches=0)

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


def list_field_ids(dataset_dir, phase_channel="C1"):
    dataset_dir = Path(dataset_dir)
    return sorted(
        {
            path.stem.rsplit("_", 1)[0]
            for path in dataset_dir.glob(f"*_{phase_channel}.tiff")
        }
    )


def plot_multichannel_overview(dataset_dir, field_id, channel_labels, figsize=(14, 3.8)):
    """Return aligned display crops for each channel in a field of view."""
    fig = plot_field_channels(dataset_dir, field_id, channel_labels)
    if figsize is not None:
        fig.set_size_inches(*figsize)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_mesh_overlay(image_object, cell_id, crop_size=150, verbose=False):
    cellobj = next((cell for cell in image_object.cells if cell.cell_id == cell_id), None)
    if cellobj is None:
        if verbose:
            print(f"Cell with ID {cell_id} not found.")
        return

    contour = np.asarray(cellobj.contour)
    mesh = np.asarray(cellobj.mesh)
    midline = np.asarray(cellobj.midline)

    center_x = contour[:, 1].mean()
    center_y = contour[:, 0].mean()
    half_crop = crop_size / 2
    xmin = center_x - half_crop
    xmax = center_x + half_crop
    ymin = center_y - half_crop
    ymax = center_y + half_crop

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(image_object.image, cmap="gray")
    ax.plot(contour[:, 1], contour[:, 0], color="white", lw=2, label="Cell contour")
    if mesh.size:
        ax.plot(
            [mesh[:, 1][::2], mesh[:, 3][::2]],
            [mesh[:, 0][::2], mesh[:, 2][::2]],
            color="cyan",
            lw=1,
        )
    ax.plot(midline[:, 1], midline[:, 0], color="magenta", lw=2, label="Midline")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Mesh overlay for cell {cell_id}")
    ax.axis("off")
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.show()


def pick_detected_cell(image_object, preferred_channels=("C4", "C5")):
    """Choose an interior, isolated cell with a detected object when possible."""
    candidates = []
    all_centers = []
    for cell in image_object.cells:
        contour = np.asarray(cell.contour)
        if contour.ndim != 2 or contour.shape[1] != 2 or not len(contour):
            continue
        center = contour.mean(axis=0)
        all_centers.append(center)
        detections = sum(
            len(contours) for channel in preferred_channels
            for contours in [cell.object_meshdata.get(channel, {}).get("object_contour")]
            if contours is not None
        )
        if detections:
            margin = min(center[0], center[1],
                         image_object.image.shape[0] - center[0],
                         image_object.image.shape[1] - center[1])
            length = np.ptp(contour, axis=0).max()
            candidates.append((cell.cell_id, center, margin, length))
    if not candidates:
        return pick_representative_cell([image_object])[1]
    median_length = np.median([item[3] for item in candidates])
    tree = cKDTree(all_centers) if len(all_centers) > 1 else None
    def score(item):
        distance = tree.query(item[1], k=2)[0][1] if tree is not None else np.inf
        return (item[2] >= 60, distance / max(item[3], 1),
                -abs(item[3] - median_length), item[2])
    return max(candidates, key=score)[0]


def plot_morphology_summary(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].hist(df["cell_length"].dropna(), bins=30, color="#4c72b0", alpha=0.9)
    axes[0, 0].set_title("Cell length distribution")
    axes[0, 0].set_xlabel("Length (um)")
    axes[0, 0].set_ylabel("Cell count")

    axes[0, 1].hist(df["cell_width"].dropna(), bins=30, color="#55a868", alpha=0.9)
    axes[0, 1].set_title("Cell width distribution")
    axes[0, 1].set_xlabel("Width (um)")
    axes[0, 1].set_ylabel("Cell count")

    scatter = axes[1, 0].scatter(
        df["cell_length"],
        df["cell_area"],
        c=df["C5_cell_total_obj_area"].fillna(0),
        cmap="magma",
        s=20,
        alpha=0.75,
    )
    axes[1, 0].set_title("Cell area vs. cell length")
    axes[1, 0].set_xlabel("Length (um)")
    axes[1, 0].set_ylabel("Area (um^2)")
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label("Nucleoid object area")

    image_names = sorted(df["image_name"].dropna().unique())
    width_groups = [
        df.loc[df["image_name"] == image_name, "cell_width"].dropna()
        for image_name in image_names
    ]
    axes[1, 1].boxplot(width_groups, labels=image_names, patch_artist=True)
    axes[1, 1].set_title("Cell width by field of view")
    axes[1, 1].set_ylabel("Width (um)")
    axes[1, 1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


def plot_signal_summary(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(
        df["C2_signal_constriction_degree_mesh"].dropna(),
        bins=30,
        color="#dd8452",
        alpha=0.9,
    )
    axes[0].set_title("Outer-membrane signal localization")
    axes[0].set_xlabel("C2 signal constriction degree")
    axes[0].set_ylabel("Cell count")

    max_objects = int(
        np.nanmax([df["C4_object_number"].max(), df["C5_object_number"].max()])
    )
    bins = np.arange(0.5, max_objects + 1.6, 1)
    axes[1].hist(
        df["C4_object_number"].dropna(),
        bins=bins,
        alpha=0.7,
        label="RNA objects",
        color="#55a868",
    )
    axes[1].hist(
        df["C5_object_number"].dropna(),
        bins=bins,
        alpha=0.7,
        label="Nucleoid objects",
        color="#c44e52",
    )
    axes[1].set_title("Detected object counts per cell")
    axes[1].set_xlabel("Objects per cell")
    axes[1].set_ylabel("Cell count")
    axes[1].legend(frameon=False)

    scatter = axes[2].scatter(
        df["C4_NC_ratio"],
        df["C5_NC_ratio"],
        c=df["cell_length"],
        cmap="viridis",
        s=20,
        alpha=0.75,
    )
    axes[2].set_title("RNA versus nucleoid occupancy")
    axes[2].set_xlabel("C4 object area / cell area")
    axes[2].set_ylabel("C5 object area / cell area")
    cbar = plt.colorbar(scatter, ax=axes[2])
    cbar.set_label("Cell length (um)")

    plt.tight_layout()
    plt.show()


def plot_pairwise_metric_heatmap(
    df,
    metric_suffix,
    channel_order=("C2", "C3", "C4", "C5"),
):
    matching_columns = [column for column in df.columns if column.endswith(metric_suffix)]
    if not matching_columns:
        print(f"No columns found for metric suffix: {metric_suffix}")
        return

    matrix = pd.DataFrame(np.nan, index=channel_order, columns=channel_order, dtype=float)
    for column in matching_columns:
        ch1, ch2, _ = column.split("_", 2)
        value = df[column].median(skipna=True)
        matrix.loc[ch1, ch2] = value
        matrix.loc[ch2, ch1] = value

    if "correlation" in metric_suffix:
        np.fill_diagonal(matrix.values, 1.0)
        vmin, vmax, cmap = -1, 1, "coolwarm"
    else:
        vmin = np.nanmin(matrix.values)
        vmax = np.nanmax(matrix.values)
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(channel_order)))
    ax.set_yticks(range(len(channel_order)))
    ax.set_xticklabels(channel_order)
    ax.set_yticklabels(channel_order)
    ax.set_title(metric_suffix.replace("_", " "))

    for row in range(len(channel_order)):
        for col in range(len(channel_order)):
            value = matrix.iloc[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="black")

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
