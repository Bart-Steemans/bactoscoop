from datetime import datetime
from pathlib import Path
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import bactoscoop
import bactoscoop.plot as bplt


# Set SOURCE_DIR to your extracted 5_channel_example folder if it is elsewhere.
SOURCE_DIR = None
if SOURCE_DIR is None:
    examples_dir = os.environ.get("BACTOSCOOP_EXAMPLES_DIR")
    candidates = ([Path(examples_dir)] if examples_dir else
                  [folder for base in (Path.cwd(), *Path.cwd().parents)
                   for folder in (base / "examples", base)])
    SOURCE_DIR = next((base / "5_channel_example" for base in candidates
                       if (base / "5_channel_example").is_dir()), None)
if SOURCE_DIR is None:
    raise FileNotFoundError("Set SOURCE_DIR to the extracted 5_channel_example folder.")
SOURCE_DIR = Path(SOURCE_DIR).expanduser().resolve()
RUN_DIR = (Path.home() / "bactoscoop_runs" /
           datetime.now().strftime("%Y%m%d_%H%M%S_%f") / "5_channel_example")
shutil.copytree(SOURCE_DIR, RUN_DIR)

PHASE_CHANNEL = "C1"
CHANNEL_LABELS = {'C1': 'Phase contrast', 'C2': 'Outer membrane', 'C3': 'Inner membrane', 'C4': 'RNA', 'C5': 'DAPI / nucleoid'}
FIELD_IDS = bplt.list_field_ids(RUN_DIR, phase_channel=PHASE_CHANNEL)
print("Source dataset:", SOURCE_DIR)
print("Working copy and outputs:", RUN_DIR)
print("Fields:", FIELD_IDS)

OVERVIEW_FIELD = FIELD_IDS[0]
OVERVIEW_CROP_PX = 650
fig = bplt.plot_field_channels(
    RUN_DIR, OVERVIEW_FIELD, CHANNEL_LABELS, crop_size=OVERVIEW_CROP_PX,
)
display(fig)
plt.close(fig)

PIXEL_SIZE_UM = 0.065
RUN_SEGMENTATION = False

ic = bactoscoop.ImageCollection(str(RUN_DIR), px=PIXEL_SIZE_UM)
ic.load_phase_images(phase_channel=PHASE_CHANNEL)
if RUN_SEGMENTATION:
    ok = ic.segment_images(
        mask_thresh=1, minsize=100, n=range(len(ic.images)),
        model_name="bact_phase_omni",
    )
    if not ok:
        raise RuntimeError(ic.processing_summary())
ic.create_image_objects(phase_channel=PHASE_CHANNEL)

SEGMENTATION_FIELD = FIELD_IDS[0]
SEGMENTATION_LABEL = None  # Set an integer label to centre on that cell.
SEGMENTATION_CENTER = None  # Optional (row, column) when label is None.
SEGMENTATION_CROP_PX = 700

review_image = next(image for image in ic.image_objects
                    if image.image_name.rsplit("_", 1)[0] == SEGMENTATION_FIELD)
fig = bplt.plot_segmentation_review(
    [review_image], crop_size=SEGMENTATION_CROP_PX,
    mask_label=SEGMENTATION_LABEL, center=SEGMENTATION_CENTER,
)
display(fig)
plt.close(fig)

LOAD_SAVED_MESHES = False
if LOAD_SAVED_MESHES:
    ic.batch_load_mesh("5_channel_example_meshdata.pkl", phase_channel=PHASE_CHANNEL)
else:
    ic.batch_process_mesh(
        phase_channel=PHASE_CHANNEL, join_thresh=4, split_thresh=0.5,
        CD_width=False, smoothing=0.1, save_data=True,
    )
print("Cells with meshes:", sum(len(image.cells) for image in ic.image_objects))

MESH_FIELD = FIELD_IDS[0]
MESH_CELL_ID = 43
MESH_WINDOW_PX = 100
mesh_image, mesh_cell_id = bplt.select_example_cell(
    ic.image_objects, field_id=MESH_FIELD, cell_id=MESH_CELL_ID,
)
fig = bplt.plot_cell_gallery(
    mesh_image, mesh_cell_id, channel=None, window_px=MESH_WINDOW_PX,
)
display(fig)
plt.close(fig)

RUN_CURATION = True
MODEL_PATH = RUN_DIR / "keio_collection_stationary.pkl"

if RUN_CURATION:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    before_contours = bplt.snapshot_cell_contours(ic.image_objects)
    before_count = sum(len(image.cells) for image in ic.image_objects)
    ic.curate_dataset(str(MODEL_PATH), control=False, save_curated_data=True)
    after_count = sum(len(image.cells) for image in ic.image_objects)
    print(f"SVM curation: {before_count:,} to {after_count:,} cells")
    fig = bplt.plot_curation_review(before_contours, ic.image_objects)
    display(fig)
    plt.close(fig)
else:
    print("SVM curation skipped; continuing with all reconstructed cells.")

detections_c2 = ic.batch_detect_objects(
    channels=["C2"], reset_channels=True, align=True,
    smoothing=0.1, log_sigma=3, kernel_width=3,
    min_overlap_ratio=0.001, max_external_ratio=0.4,
)
detections_c345 = ic.batch_detect_objects(
    channels=["C3", "C4", "C5"], reset_channels=False, align=False,
    smoothing=0.1, log_sigma=3, kernel_width=3,
    min_overlap_ratio=0.001, max_external_ratio=0.3,
)
detections = pd.concat([detections_c2, detections_c345], ignore_index=True)
object_counts = (detections.assign(object_count=detections["object_contours"].map(len))
                 .groupby("channel")["object_count"].sum())
print("Detected objects per channel:")
display(object_counts)

ALIGNMENT_FIELD = FIELD_IDS[0]
ALIGNMENT_CELL_ID = 43
ALIGNMENT_WINDOW_PX = 110
alignment_image, alignment_cell_id = bplt.select_example_cell(
    ic.image_objects, field_id=ALIGNMENT_FIELD, cell_id=ALIGNMENT_CELL_ID,
)
fig = bplt.plot_C2_alignment(
    alignment_image, cell_id=alignment_cell_id, window_px=ALIGNMENT_WINDOW_PX,
    channel="C2", channel_label="Outer membrane", show_objects=True,
)
display(fig)
plt.close(fig)

OBJECT_FIELD = FIELD_IDS[0]
OBJECT_CELL_ID = 43
OBJECT_WINDOW_PX = 100
object_image, object_cell_id = bplt.select_example_cell(
    ic.image_objects, field_id=OBJECT_FIELD, cell_id=OBJECT_CELL_ID,
)

fig = bplt.plot_cell_gallery(
    object_image, object_cell_id, channel="C2", channel_label="Outer membrane",
    align_signal=True, show_objects=True, window_px=OBJECT_WINDOW_PX,
)
display(fig)
plt.close(fig)
fig = bplt.plot_signal_gallery(
    object_image, object_cell_id,
    channel_labels={"C3": "Inner membrane", "C4": "RNA", "C5": "DAPI / nucleoid"},
    window_px=OBJECT_WINDOW_PX, show_objects=True,
)
display(fig)
plt.close(fig)

# First pass: locally aligned outer-membrane signal.
ic.batch_calculate_features(
    [(["C2"], "membrane"), (["C2"], "profiling"), (["C2"], "objects")],
    all_data=False, reset=True, shift_signal=True, max_mesh_size=1000,
)
# Second pass: morphology and the remaining channels.
ic.batch_calculate_features(
    [
        ([None], "morphological"),
        (["C3"], "membrane"),
        (["C3", "C4", "C5"], "profiling"),
        (["C3", "C4", "C5"], "objects"),
    ],
    all_data=False, reset=False, shift_signal=False, max_mesh_size=1000,
    retain_contour_on_object_mesh_failure_channels=["C4", "C5"],
)
features = ic.merge_dataframes(discard_morphological_nan=True)
print(f"{len(features):,} cells × {len(features.columns)} columns")
display(features[["image_name", "cell_id", "cell_length", "cell_width", "cell_area"]].head())

from matplotlib.ticker import MaxNLocator

HISTOGRAM_BINS = 25
plot_features = [
    ("cell_length", "Cell length (µm)"),
    ("cell_width", "Cell width (µm)"),
    ("cell_area", "Cell area (µm²)"),
    ("C5_object_number", "DAPI objects per cell"),
]
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for ax, (column, label) in zip(axes.flat, plot_features):
    values = features[column].dropna()
    bins = ([i - 0.5 for i in range(int(values.max()) + 2)]
            if column.endswith("object_number") and not values.empty else HISTOGRAM_BINS)
    ax.hist(values, bins=bins, color="#278C9E", edgecolor="white")
    ax.set(xlabel=label, ylabel="Cells")
    if column.endswith("object_number"):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(left=-0.5)
fig.tight_layout()
display(fig)
plt.close(fig)

PLOT_AXIAL_PROFILES = False
PROFILE_FIELD = FIELD_IDS[0]
PROFILE_CELL_ID = 43
PROFILE_CHANNELS = ["C2", "C3", "C4", "C5"]
if PLOT_AXIAL_PROFILES:
    profile_image, profile_cell_id = bplt.select_example_cell(
        ic.image_objects, field_id=PROFILE_FIELD, cell_id=PROFILE_CELL_ID,
    )
    figures = bplt.plot_normalized_axial_intensity(
        features, channels=PROFILE_CHANNELS, selected_frame=profile_image.frame,
        selected_cell_id=profile_cell_id, show=False,
    )
    for fig in figures:
        display(fig)
        plt.close(fig)

APPLY_FEATURE_FILTER = False
MIN_CELL_LENGTH_UM = 1.0
MAX_CELL_LENGTH_UM = 10.0

if APPLY_FEATURE_FILTER:
    keep = features["cell_length"].between(MIN_CELL_LENGTH_UM, MAX_CELL_LENGTH_UM)
    filtered_features = features.loc[keep].copy()
else:
    filtered_features = features.copy()
print(f"Feature-table selection: {len(features):,} to {len(filtered_features):,} cells")

ic.dataframe_to_parquet()
pickle_path = ic.dataframe_to_pkl()
if APPLY_FEATURE_FILTER:
    filtered_features.to_parquet(
        RUN_DIR / "5_channel_example_filtered_features.parquet", index=False,
    )
saved = pd.read_parquet(RUN_DIR / "5_channel_example_features.parquet")
print("Saved full table:", len(saved), "rows")
print("Outputs:", RUN_DIR)
