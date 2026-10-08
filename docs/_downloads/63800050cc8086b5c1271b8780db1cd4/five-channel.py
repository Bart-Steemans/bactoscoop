from pathlib import Path
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Import BactoScoop
# Import the package installed in this notebook kernel.
import bactoscoop
import bactoscoop.plot as bplt


# Paths
# Locate the downloaded examples; keep the supplied inputs unchanged.
import os
import shutil
from datetime import datetime

EXAMPLES_DIR = os.environ.get("BACTOSCOOP_EXAMPLES_DIR")
candidates = ([Path(EXAMPLES_DIR)] if EXAMPLES_DIR else
              [candidate for base in (Path.cwd(), *Path.cwd().parents)
               for candidate in (base / "examples", base)])
SOURCE_DIR = next((base / "5_channel_example" for base in candidates
                   if (base / "5_channel_example").is_dir()), None)
if SOURCE_DIR is None:
    raise FileNotFoundError("Open this notebook from the downloaded examples folder, "
                            "or set BACTOSCOOP_EXAMPLES_DIR to that folder.")
RUN_DIR = Path.home() / "bactoscoop_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f") / "5_channel_example"
shutil.copytree(SOURCE_DIR, RUN_DIR)
MODEL_PATH = RUN_DIR / "keio_collection_stationary.pkl"

# Settings
RUN_SEGMENTATION = False  # Reuse supplied masks; set True for fresh Omnipose inference.
PIXEL_SIZE_UM = 0.065
PHASE_CHANNEL = "C1"

EXAMPLE_FIELD = "Sample_E4_XY4"
EXAMPLE_CELL_ID = 30

# Channels
OUTER_MEMBRANE = "C2"
INNER_MEMBRANE = "C3"
RNA_CHANNEL = "C4"
DAPI_CHANNEL = "C5"

SIGNAL_CHANNELS = [
    OUTER_MEMBRANE, INNER_MEMBRANE, RNA_CHANNEL, DAPI_CHANNEL
]

CHANNEL_LABELS = {
    "C1": "Phase contrast",
    "C2": "Outer membrane",
    "C3": "Inner membrane",
    "C4": "RNA",
    "C5": "DAPI / nucleoid",
}

# Validate paths
if not RUN_DIR.is_dir():
    raise FileNotFoundError(RUN_DIR)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)

(RUN_DIR / "masks").mkdir(exist_ok=True)

FIELD_IDS = bplt.list_field_ids(RUN_DIR, phase_channel=PHASE_CHANNEL)

print("Fields:", FIELD_IDS)
print("Image directory:", RUN_DIR)
print("CUDA available:", torch.cuda.is_available())
print("Working copy and outputs:", RUN_DIR)
print("CUDA available:", torch.cuda.is_available(), "(segmentation uses CPU when False)")

OUTER_MEMBRANE = "C2"
INNER_MEMBRANE = "C3"
RNA_CHANNEL = "C4"
DAPI_CHANNEL = "C5"
SIGNAL_CHANNELS = [OUTER_MEMBRANE, INNER_MEMBRANE, RNA_CHANNEL, DAPI_CHANNEL]
CHANNEL_LABELS = {"C1": "Phase contrast", "C2": "Outer membrane", "C3": "Inner membrane",
                  "C4": "RNA", "C5": "DAPI / nucleoid"}

fig = bplt.plot_multichannel_overview(RUN_DIR, FIELD_IDS[1], CHANNEL_LABELS)
display(fig)
plt.close(fig)

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
SEGMENTATION_FIELD = None
SEGMENTATION_LABEL = None
SEGMENTATION_CENTER = None
SEGMENTATION_CROP_PX = 700
review_field = SEGMENTATION_FIELD or FIELD_IDS[0]
review_image = next(image for image in ic.image_objects
                    if image.image_name.rsplit("_", 1)[0] == review_field)
fig = bplt.plot_segmentation_review(
    [review_image], crop_size=SEGMENTATION_CROP_PX,
    mask_label=SEGMENTATION_LABEL, center=SEGMENTATION_CENTER,
)
display(fig)
plt.close(fig)

ic.batch_process_mesh(
    phase_channel=PHASE_CHANNEL, join_thresh=4, split_thresh=0.5,
    CD_width=False, smoothing=0.1, save_data=True,
)
print(f"Meshes: {sum(len(image.cells) for image in ic.image_objects):,}")

# Load the save mesh data from the previous run
ic.batch_load_mesh("5_channel_example_meshdata.pkl", phase_channel=PHASE_CHANNEL)

fig = bplt.plot_cell_gallery(image_object=ic.image_objects[0],cell_id=43, channel=None, window_px=100)
display(fig)
plt.close(fig)

before_contours = bplt.snapshot_cell_contours(ic.image_objects)
before_count = sum(len(image.cells) for image in ic.image_objects)
ic.curate_dataset(str(MODEL_PATH), control=False, save_curated_data=True)
after_count = sum(len(image.cells) for image in ic.image_objects)
print(f"SVM curation: {before_count:,} to {after_count:,} cells")
fig = bplt.plot_curation_review(before_contours, ic.image_objects)
display(fig)
plt.close(fig)

detections_c2 = ic.batch_detect_objects(
    channels=[OUTER_MEMBRANE], reset_channels=True, smoothing=0.1,
    align=True, log_sigma=3, kernel_width=3,
    min_overlap_ratio=0.001, max_external_ratio=0.4,
)
detections_c345 = ic.batch_detect_objects(
    channels=[INNER_MEMBRANE, RNA_CHANNEL, DAPI_CHANNEL],
    reset_channels=False, align=False, smoothing=0.1, log_sigma=3,
    kernel_width=3, min_overlap_ratio=0.001, max_external_ratio=0.3,
)
print("Detected objects by channel:")
display(pd.concat([detections_c2, detections_c345])["channel"].value_counts())

fig = bplt.plot_C2_alignment(
    ic.image_objects[0],
    cell_id=43,
    window_px=110,
    channel="C2",
    channel_label="Outer membrane",
    show_objects=True,
)
display(fig)
plt.close(fig)

membrane_field = ic.image_objects[0].image_name.rsplit("_", 1)[0]
membrane_image = next(image for image in ic.image_objects
                      if image.image_name.rsplit("_", 1)[0] == membrane_field)
membrane_cell_id = 43
selected_membrane_cell = next(
    (cell for cell in membrane_image.cells if cell.cell_id == membrane_cell_id), None
)
if selected_membrane_cell is None or not selected_membrane_cell.object_meshdata.get(
    OUTER_MEMBRANE, {}
).get("object_contour"):
    print("Selected C2 cell has no detected object; choosing one with a detection.")
    membrane_cell_id = bplt.pick_detected_cell(
        membrane_image, preferred_channels=(OUTER_MEMBRANE,)
    )
print("Aligned C2 example:", membrane_image.image_name, membrane_cell_id)
fig = bplt.plot_cell_gallery(
    membrane_image, membrane_cell_id, channel=OUTER_MEMBRANE,
    channel_label="Outer membrane", align_signal=True, show_objects=True, window_px=100,
)
display(fig)
plt.close(fig)

fig = bplt.plot_signal_gallery(
    ic.image_objects[0], 43,
    channel_labels={RNA_CHANNEL: "RNA", DAPI_CHANNEL: "DAPI / nucleoid"},
    window_px=100, show_objects=True,
)
display(fig)
plt.close(fig)

# First pass: outer membrane (signal-shift correction enabled)
ic.batch_calculate_features(
    [
        ([OUTER_MEMBRANE], "membrane"),   # Measure membrane-associated features
        ([OUTER_MEMBRANE], "profiling"),  # Measure intensity profiles across the cell
        ([OUTER_MEMBRANE], "objects"),    # Calculate features of detected objects
    ],
    all_data=False,      # Pass False to the feature methods; its effect is method-specific
    reset=True,          # Clear previously stored feature DataFrames before this pass
    shift_signal=True,   # Enable signal-shift handling for these measurements
    max_mesh_size=1000,  # Exclude cells exceeding the maximum permitted mesh size
)

# Second pass: morphology, inner membrane, RNA and DAPI
ic.batch_calculate_features(
    [
        ([None], "morphological"),  # Cell morphology: length, width, area, volume, etc.
        ([INNER_MEMBRANE], "membrane"),  # Inner-membrane-associated measurements
        ([INNER_MEMBRANE, RNA_CHANNEL, DAPI_CHANNEL], "profiling"),  # Signal intensity profiles
        ([INNER_MEMBRANE, RNA_CHANNEL, DAPI_CHANNEL], "objects"),    # Detected-object features
    ],
    all_data=False,      # Same feature-output setting as the first pass
    reset=False,         # Preserve the outer-membrane feature DataFrames from the first pass
    shift_signal=False,  # Do not apply signal-shift handling in this pass
    max_mesh_size=1000,  # Apply the same maximum mesh-size threshold
    retain_contour_on_object_mesh_failure_channels=[
        RNA_CHANNEL,
        DAPI_CHANNEL,
    ],  # For these channels, retain contour-based object measurements
        # when detected-object mesh construction fails, rather than
        # omitting the affected cell's objects-feature row
)

# Combine the feature DataFrames into one table
analysis_df = ic.merge_dataframes(
    include_metadata_tag=False,       # Do not add the metadata tag to the merged output
    discard_morphological_nan=True,   # Exclude rows failing the merge's morphological NaN checks
)

# Inspect the result
print(f"{len(analysis_df):,} cells × {len(analysis_df.columns)} columns")

display(
    analysis_df[
        ["image_name", "cell_id", "cell_length", "cell_width", "cell_area"]
    ].head()
)

fig = bplt.plot_feature_overview(analysis_df, signal_column="C4_NC_ratio")
display(fig)
plt.close(fig)
bplt.plot_signal_summary(analysis_df)
profile_figures = bplt.plot_normalized_axial_intensity(
    analysis_df, channels=SIGNAL_CHANNELS,
    selected_frame=ic.image_objects[0].frame, selected_cell_id=43,
    show=False,
)
for fig in profile_figures:
    display(fig)
    plt.close(fig)

ic.dataframe_to_parquet()
pickle_path = ic.dataframe_to_pkl()
saved = pd.read_parquet(RUN_DIR / "5_channel_example_features.parquet")
print("Saved Parquet and pickle:", len(saved), "rows;", pickle_path)
