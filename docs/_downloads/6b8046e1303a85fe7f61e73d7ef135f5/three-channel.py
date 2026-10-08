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

# Point to the folder containing C1-C3 TIFFs, masks/, and the SVM model.
# Locate the downloaded examples; keep the supplied inputs unchanged.
import os
import shutil
from datetime import datetime

EXAMPLES_DIR = os.environ.get("BACTOSCOOP_EXAMPLES_DIR")
candidates = ([Path(EXAMPLES_DIR)] if EXAMPLES_DIR else
              [candidate for base in (Path.cwd(), *Path.cwd().parents)
               for candidate in (base / "examples", base)])
SOURCE_DIR = next((base / "3_channel_example" for base in candidates
                   if (base / "3_channel_example").is_dir()), None)
if SOURCE_DIR is None:
    raise FileNotFoundError("Open this notebook from the downloaded examples folder, "
                            "or set BACTOSCOOP_EXAMPLES_DIR to that folder.")
RUN_DIR = Path.home() / "bactoscoop_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f") / "3_channel_example"
shutil.copytree(SOURCE_DIR, RUN_DIR)
MODEL_PATH = RUN_DIR / "DnaN_Timecourse.pkl"
RUN_SEGMENTATION = False  # Reuse supplied masks; set True for fresh Omnipose inference.
PIXEL_SIZE_UM = 0.065  # Replace with your microscope calibration.
PHASE_CHANNEL, DAPI, GFP_DNAN = "C1", "C2", "C3"
CHANNEL_LABELS = {"C1": "Phase contrast", "C2": "DAPI", "C3": "GFP-DnaN"}

if not RUN_DIR.is_dir():
    raise FileNotFoundError(f"Example folder not found: {RUN_DIR}")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"SVM model not found: {MODEL_PATH}")
(RUN_DIR / "masks").mkdir(exist_ok=True)
FIELD_IDS = bplt.list_field_ids(RUN_DIR, phase_channel=PHASE_CHANNEL)
print("Fields:", FIELD_IDS)
print("Image and output folder:", RUN_DIR)
print("CUDA available:", torch.cuda.is_available(), "(CPU is used when False)")

fig = bplt.plot_multichannel_overview(RUN_DIR, FIELD_IDS[1], CHANNEL_LABELS, )
display(fig)
plt.close(fig)

collection = bactoscoop.ImageCollection(str(RUN_DIR), px=PIXEL_SIZE_UM)
collection.load_phase_images(phase_channel=PHASE_CHANNEL)
if RUN_SEGMENTATION:
    ok = collection.segment_images(
        mask_thresh=1, minsize=100, n=range(len(collection.images)),
        model_name="bact_phase_omni",
    )
    if not ok:
        raise RuntimeError(collection.processing_summary())
collection.create_image_objects(phase_channel=PHASE_CHANNEL)

# Change these values and rerun this cell to inspect another field or label.
SEGMENTATION_FIELD = FIELD_IDS[0]
SEGMENTATION_LABEL = None  # Integer mask label, or None for a central cell.
SEGMENTATION_CENTER = None  # Optional (row, column), used when label is None.
SEGMENTATION_CROP_PX = 700
review_image = next(image for image in collection.image_objects
                    if image.image_name.rsplit("_", 1)[0] == SEGMENTATION_FIELD)
fig = bplt.plot_segmentation_review(
    [review_image], crop_size=SEGMENTATION_CROP_PX,
    mask_label=SEGMENTATION_LABEL, center=SEGMENTATION_CENTER,
)
display(fig)
plt.close(fig)

collection.batch_process_mesh(
    phase_channel=PHASE_CHANNEL, join_thresh=4, split_thresh=0.5,
    CD_width=False, smoothing=0.1, save_data=True,
)
print("Cells with meshes:", sum(len(image.cells) for image in collection.image_objects))

# Uncomment on a later run after skipping mesh construction:
# collection.batch_load_mesh("3_channel_example_meshdata.pkl", phase_channel=PHASE_CHANNEL)

MESH_FIELD = "TL4_SGL711_M9glu_12H_XY5"
MESH_CELL_ID = 30
mesh_image, _ = bplt.select_example_cell(
    collection.image_objects, field_id=MESH_FIELD, cell_id=MESH_CELL_ID,
)
fig = bplt.plot_cell_gallery(mesh_image, MESH_CELL_ID, channel=None)
display(fig)
plt.close(fig)

#collection.batch_load_mesh("3_channel_example_meshdata.pkl", phase_channel=PHASE_CHANNEL)
before = bplt.snapshot_cell_contours(collection.image_objects)
before_count = sum(len(image.cells) for image in collection.image_objects)
collection.curate_dataset(str(MODEL_PATH), control=False, save_curated_data=True)
after_count = sum(len(image.cells) for image in collection.image_objects)
print(f"SVM curation: {before_count:,} to {after_count:,} cells")
fig = bplt.plot_curation_review(before, collection.image_objects
)
display(fig)
plt.close(fig)

detections = collection.batch_detect_objects(
    channels=[DAPI,GFP_DNAN], reset_channels=True, align=False,
    smoothing=0.1, log_sigma=3, kernel_width=3,
    min_overlap_ratio=0.001, max_external_ratio=0.5,
)
print("Detected C3 objects:", len(detections))
collection.add_channels(collection.image_objects, [DAPI, GFP_DNAN])

OBJECT_FIELD = "TL4_SGL711_M9glu_12H_XY5"
OBJECT_CELL_ID = 62
object_image, _ = bplt.select_example_cell(
    collection.image_objects, field_id=OBJECT_FIELD, cell_id=OBJECT_CELL_ID,
)
fig = bplt.plot_cell_gallery(
    object_image,30, channel=DAPI, channel_label="DAPI",
)
display(fig)
plt.close(fig)
fig = bplt.plot_signal_gallery(
    object_image, 30,
    channel_labels={DAPI: "DAPI", GFP_DNAN: "GFP-DnaN"},
    show_objects=True,
)
display(fig)
plt.close(fig)

channel_method_pairs = [
    ([None], "morphological"),
    ([DAPI], "profiling"),
    ([GFP_DNAN], "profiling"),
    ([GFP_DNAN], "objects"),
]
collection.batch_calculate_features(
    channel_method_pairs, all_data=False, max_mesh_size=1000,
)
features = collection.merge_dataframes()
print(f"{len(features):,} cells x {len(features.columns)} columns")
display(features[["image_name", "cell_id", "cell_volume", "C3_NC_ratio"]].head())

fig, ax = plt.subplots(figsize=(5.5, 4))
valid = features[["cell_volume", "C3_NC_ratio"]].dropna()
ax.scatter(valid.cell_volume, valid.C3_NC_ratio,
           s=12, alpha=0.45, color="#31A85E")
ax.set(xlabel="Cell volume (um^3)", ylabel="C3 object area / cell area",
       title="GFP-DnaN object fraction vs cell volume")
fig.tight_layout()
display(fig)
plt.close(fig)

collection.dataframe_to_parquet()
pickle_path = collection.dataframe_to_pkl()
saved = pd.read_parquet(RUN_DIR / "3_channel_example_features.parquet")
summary = collection.processing_summary()
print("Saved Parquet and pickle:", len(saved), "rows;", pickle_path)
print("Image errors:", summary["error_count"],
      "Cell errors:", summary["cell_error_count"])
