# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:00:33 2024

@author: Bart Steemans. Govers Lab.
"""

import os
# Run: C:\Users\Nikon\anaconda3\envs\bactoscoop\python.exe "C:\Users\Bart\bactoscoop\paralell processing keio collection bactoscoop_v.260330.py"
# in command line

# Keep numerical libraries single-threaded inside each worker so the outer
# multiprocessing pool remains the only concurrency layer.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Keep bactoscoop batch runs quiet enough to monitor.
os.environ.setdefault("BACTOSCOOP_LOG_LEVEL", "WARNING")

import sys
import glob
import json
import traceback
import multiprocessing as mp
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from pathlib import Path
from time import perf_counter

parent_dir = "C:/Users/Bart/bactoscoop/"
sys.path.append(parent_dir)

import bactoscoop


CHANNEL1 = "C1"
CHANNEL2 = "C2"
CHANNEL3 = "C3"
CHANNEL4 = "C4"
CHANNEL5 = "C5"
EXPORT_CURATED_MASKS = False
CURATED_MASK_SUBFOLDER = "curated_masks"

CURATION_DATASET = (
    "C:/Users/Bart/250821_randompoles_drive1_drive2_drive3_keio_collection.pkl"
)
TOP_LEVEL_FOLDERS = [
    r"F:\Export Bart\Stationary_phase_screen",
    r"I:\Export Bart\Stationary_phase_screen",
    r"K:\Export Bart\Stationary_phase_screen",
    r"J:\Stationary_phase_screen",
]
DEFAULT_CORES = 12
MAX_TASKS_PER_CHILD = 4
PROGRESS_DIR = Path(parent_dir) / "run_status"


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        yield


def print_stage(folder_name, stage, message):
    print(f"[{folder_name}] {stage} | {message}", flush=True)


def process_subfolder(folder_path):
    folder_path = str(folder_path)
    folder_name = os.path.basename(os.path.normpath(folder_path))
    started = perf_counter()
    ic = None

    try:

        ic = bactoscoop.ImageCollection(folder_path)

        # Segmentation -------------------------------------------------------------------------------------------------------------------------------------
        ic.load_phase_images(phase_channel=CHANNEL1)
        n = len(ic.images)
        ic.segment_images(n = range(n), 
                      minsize = 100, # minimum size of masks in pixels
                      mask_thresh = 1) # Increasing this parameter will make the mask smaller, decreasing bigger
        ic.create_image_objects(phase_channel=CHANNEL1)
        image_count = len(ic.image_objects)
        with suppress_stdout():
            ic.batch_process_mesh(
                object_list=None,
                phase_channel=CHANNEL1,
                join_thresh=4,
                split_thresh=0.5,
                CD_width=False,
                smoothing=0.1,
                save_data=True,
                neighbor_filter_max_neighbors=3,
                neighbor_filter_connectivity=1,
            )
        mesh_rows = len(ic.mesh_df_collection)
        prefilter_removed_labels = 0
        if (
            getattr(ic, "mesh_prefilter_stats", None) is not None
            and not ic.mesh_prefilter_stats.empty
        ):
            prefilter_removed_labels = int(
                ic.mesh_prefilter_stats["removed_label_count"].sum()
            )
        print_stage(
            folder_name,
            "mesh",
            f"images={image_count} | meshes={mesh_rows} | prefilter_removed={prefilter_removed_labels}",
        )

        ic.batch_load_mesh(folder_name + "_meshdata.pkl", phase_channel=CHANNEL1)
        with suppress_stdout():
            ic.curate_dataset(CURATION_DATASET, control=False)
        curated_mesh_rows = len(ic.mesh_df_collection)
        curated_cell_count = sum(len(image.cells) for image in ic.image_objects)
        curated_mask_count = 0
        if EXPORT_CURATED_MASKS:
            curated_mask_paths, curated_mask_provenance = ic.export_curated_masks(
                output_subfolder=CURATED_MASK_SUBFOLDER,
                overwrite=True,
            )
            curated_mask_count = len(curated_mask_paths)
        print_stage(
            folder_name,
            "curate",
            f"curated_meshes={curated_mesh_rows} | curated_cells={curated_cell_count}",
        )
        if EXPORT_CURATED_MASKS:
            print_stage(
                folder_name,
                "curated_masks",
                f"files={curated_mask_count} | folder={CURATED_MASK_SUBFOLDER}",
            )
        ic.batch_load_mesh(folder_name + "_curated_meshdata.pkl", phase_channel=CHANNEL1)
        image_count = len(ic.image_objects)
        curated_mesh_rows = len(ic.mesh_df_collection)
        curated_cell_count = sum(len(image.cells) for image in ic.image_objects)
        print_stage(
            folder_name,
            "load_curated",
            f"images={image_count} | curated_meshes={curated_mesh_rows} | curated_cells={curated_cell_count}",
        )

        outer_detection_df = ic.batch_detect_objects(
            channels=[CHANNEL2],
            reset_channels=True,
            smoothing=0.1,
            align=True,
            log_sigma=3,
            kernel_width=3,
            min_overlap_ratio=0.001,
            max_external_ratio=0.4,
        )

        other_detection_df = ic.batch_detect_objects(
            channels=[CHANNEL3, CHANNEL4, CHANNEL5],
            reset_channels=False,
            align=False,
            smoothing=0.1,
            log_sigma=3,
            kernel_width=3,
            min_overlap_ratio=0.001,
            max_external_ratio=0.3,
        )

        channel_method_tuple1 = [
            ([CHANNEL2], "membrane"),
            ([CHANNEL2], "profiling"),
            ([CHANNEL2], "objects"),
        ]
        ic.batch_calculate_features(
            channel_method_tuple1,
            all_data=False,
            reset=True,
            shift_signal=True,
            max_mesh_size=1000
        )

        channel_method_tuple2 = [
            ([None], "morphological"),
            ([CHANNEL3], "membrane"),
            ([CHANNEL3, CHANNEL4, CHANNEL5], "profiling"),
            ([CHANNEL3, CHANNEL4, CHANNEL5], "objects"),
        ]
        ic.batch_calculate_features(
            channel_method_tuple2,
            all_data=False,
            reset=False,
            shift_signal=False,
            max_mesh_size=1000, retain_contour_on_object_mesh_failure_channels=[CHANNEL4, CHANNEL5]
        )

        ic.merge_dataframes(include_metadata_tag=True, discard_morphological_nan=True)

        feature_method_tuples = [
            (
                [
                    "normalized_axial_intensity",
                    "normalized_average_mesh_intensity",
                    "radial_intensity_distribution",
                ],
                [
                    "manders",
                    "pearson",
                    "li_icq",
                    "spearman",
                    "kendall",
                    "distance_corr",
                    "covariance",
                    "n_cross_corr",
                    "entropy_diff",
                    "kurtosis_ratio",
                    "skewness_product",
                    "zero_crossings_diff",
                    "fft_peak_ratio",
                    "fft_energy_ratio",
                    "histogram_intersection",
                    "cosine_similarity",
                ],
            ),
            (["cell_total_obj_area"], ["ratio"]),
        ]
        ic.batch_calculate_signal_correlation_features(
            ic.merged_features,
            [CHANNEL2, CHANNEL3, CHANNEL4, CHANNEL5],
            feature_method_tuples=feature_method_tuples,
        )

        feature_method_tuples = [
            (
                ["normalized_contour_intensity", "complemented_contour_intensity"],
                [
                    "manders",
                    "pearson",
                    "li_icq",
                    "spearman",
                    "kendall",
                    "distance_corr",
                    "covariance",
                    "n_cross_corr",
                    "entropy_diff",
                    "kurtosis_ratio",
                    "skewness_product",
                    "zero_crossings_diff",
                    "fft_peak_ratio",
                    "fft_energy_ratio",
                    "histogram_intersection",
                    "cosine_similarity",
                ],
            )
        ]
        ic.batch_calculate_signal_correlation_features(
            ic.merged_features,
            [CHANNEL2, CHANNEL3],
            feature_method_tuples=feature_method_tuples,
        )

        ic.dataframe_to_pkl(pkl_name="features_5")
        feature_rows = len(ic.merged_features)
        processing_summary = ic.processing_summary()
        run_status = processing_summary["status"]
        elapsed_minutes = round((perf_counter() - started) / 60, 2)
        print_stage(
            folder_name,
            "done",
            f"feature_rows={feature_rows} | elapsed={elapsed_minutes} min",
        )

        return {
            "folder_path": folder_path,
            "status": run_status,
            "processing_error_count": processing_summary["error_count"],
            "processing_errors": processing_summary["errors"],
            "completed_stages": processing_summary["completed_stages"],
            "elapsed_minutes": elapsed_minutes,
            "image_count": image_count,
            "curated_mesh_rows": curated_mesh_rows,
            "curated_cell_count": curated_cell_count,
            "feature_rows": feature_rows,
        }
    except Exception as exc:
        elapsed_minutes = round((perf_counter() - started) / 60, 2)
        processing_summary = ic.processing_summary() if ic is not None else {}
        return {
            "folder_path": folder_path,
            "status": "failed",
            "processing_error_count": processing_summary.get("error_count", 0),
            "processing_errors": processing_summary.get("errors", []),
            "completed_stages": processing_summary.get("completed_stages", []),
            "elapsed_minutes": elapsed_minutes,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def collect_pending_folders(top_level_folder):
    pending_folders = []

    first_level_folders = [
        os.path.join(top_level_folder, folder_name)
        for folder_name in os.listdir(top_level_folder)
        if os.path.isdir(os.path.join(top_level_folder, folder_name))
    ]

    for first_level in first_level_folders:
        second_level_folders = [
            os.path.join(first_level, folder_name)
            for folder_name in os.listdir(first_level)
            if os.path.isdir(os.path.join(first_level, folder_name))
        ]

        for second_level in second_level_folders:
            folder_name = os.path.basename(second_level)
            feature_files = glob.glob(
                os.path.join(second_level, f"{folder_name}_features_5.pkl")
            )
            status_path = os.path.join(second_level, "_bactoscoop_run_status.json")
            clean_success = False
            if feature_files and os.path.isfile(status_path):
                try:
                    with open(status_path, "r", encoding="utf-8") as status_file:
                        clean_success = json.load(status_file).get("status") == "completed"
                except (OSError, json.JSONDecodeError):
                    clean_success = False
            if not clean_success:
                pending_folders.append(second_level)

    return pending_folders


def progress_file_for(top_level_folder):
    drive_name = Path(top_level_folder).drive.replace(":", "")
    return PROGRESS_DIR / f"stationary_phase_screen_{drive_name}_progress.json"


def write_progress(progress_path, payload):
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(progress_path)


def process_folders_in_parallel(top_level_folder, n_cores=DEFAULT_CORES):
    pending_folders = collect_pending_folders(top_level_folder)
    total_pending = len(pending_folders)
    worker_count = min(n_cores, mp.cpu_count(), total_pending) if total_pending else 0
    progress_path = progress_file_for(top_level_folder)
    started_at = datetime.now()

    print(
        f"{top_level_folder}: found {total_pending} second-level folders missing *features_5.pkl."
    )

    if total_pending == 0:
        write_progress(
            progress_path,
            {
                "top_level_folder": top_level_folder,
                "status": "no_work",
                "worker_count": 0,
                "total_pending": 0,
                "completed": 0,
                "remaining": 0,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return {
            "top_level_folder": top_level_folder,
            "status": "no_work",
            "worker_count": 0,
            "total_pending": 0,
            "completed": 0,
            "failed": 0,
        }

    failed_results = []
    completed = 0

    write_progress(
        progress_path,
        {
            "top_level_folder": top_level_folder,
            "status": "running",
            "worker_count": worker_count,
            "total_pending": total_pending,
            "completed": 0,
            "remaining": total_pending,
            "started_at": started_at.isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    for folder_path in pending_folders:
        status_file = os.path.join(folder_path, "_bactoscoop_run_status.json")
        status_temp = status_file + ".tmp"
        with open(status_temp, "w", encoding="utf-8") as status_handle:
            json.dump(
                {
                    "folder_path": folder_path,
                    "status": "running",
                    "started_at": started_at.isoformat(timespec="seconds"),
                },
                status_handle,
                indent=2,
            )
        os.replace(status_temp, status_file)

    with mp.Pool(
        processes=worker_count,
        maxtasksperchild=MAX_TASKS_PER_CHILD,
    ) as pool:
        for result in pool.imap_unordered(process_subfolder, pending_folders, chunksize=1):
            completed += 1
            remaining = total_pending - completed
            elapsed_hours = round((datetime.now() - started_at).total_seconds() / 3600, 2)
            completion_rate = completed / max(
                (datetime.now() - started_at).total_seconds(), 1
            )
            eta_hours = (
                round((remaining / completion_rate) / 3600, 2)
                if completion_rate > 0 and remaining > 0
                else 0
            )

            status_file = os.path.join(
                result["folder_path"], "_bactoscoop_run_status.json"
            )
            status_temp = status_file + ".tmp"
            with open(status_temp, "w", encoding="utf-8") as status_handle:
                json.dump(result, status_handle, indent=2, default=str)
            os.replace(status_temp, status_file)

            if result["status"] != "completed":
                failure = {
                    "folder_path": result["folder_path"],
                    "status": result["status"],
                    "error": result.get("error"),
                    "processing_error_count": result.get("processing_error_count", 0),
                    "processing_errors": result.get("processing_errors", []),
                    "elapsed_minutes": result["elapsed_minutes"],
                }
                failed_results.append(failure)
                print(
                    f"{result['status'].upper()}: {result['folder_path']} | "
                    f"{failure['error'] or failure['processing_error_count']}"
                )

            write_progress(
                progress_path,
                {
                    "top_level_folder": top_level_folder,
                    "status": "running",
                    "worker_count": worker_count,
                    "total_pending": total_pending,
                    "completed": completed,
                    "succeeded": completed - len(failed_results),
                    "failed": len(failed_results),
                    "remaining": remaining,
                    "elapsed_hours": elapsed_hours,
                    "eta_hours": eta_hours,
                    "started_at": started_at.isoformat(timespec="seconds"),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "last_finished_folder": result["folder_path"],
                    "last_status": result["status"],
                    "last_elapsed_minutes": result["elapsed_minutes"],
                    "last_image_count": result.get("image_count"),
                    "last_curated_mesh_rows": result.get("curated_mesh_rows"),
                    "last_curated_cell_count": result.get("curated_cell_count"),
                    "last_feature_rows": result.get("feature_rows"),
                    "failed_folders": failed_results,
                },
            )

            print(
                f"[{top_level_folder}] {completed}/{total_pending} done | "
                f"ok={completed - len(failed_results)} | failed={len(failed_results)} | "
                f"remaining={remaining} | last={result['folder_path']}"
            )

    final_status = "completed_with_failures" if failed_results else "completed"
    summary = {
        "top_level_folder": top_level_folder,
        "status": final_status,
        "worker_count": worker_count,
        "total_pending": total_pending,
        "completed": completed,
        "failed": len(failed_results),
    }

    write_progress(
        progress_path,
        {
            **summary,
            "remaining": 0,
            "started_at": started_at.isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "failed_folders": failed_results,
        },
    )
    return summary


def process_drives_sequentially(top_level_folders, n_cores=DEFAULT_CORES):
    summaries = []
    for top_level_folder in top_level_folders:
        print(f"Starting drive run: {top_level_folder} | workers={n_cores}")
        summary = process_folders_in_parallel(top_level_folder, n_cores=n_cores)
        summaries.append(summary)
        print(
            f"Finished drive run: {top_level_folder} | status={summary['status']} | "
            f"completed={summary['completed']} | failed={summary['failed']}"
        )
    return summaries


if __name__ == "__main__":
    mp.freeze_support()
    process_drives_sequentially(TOP_LEVEL_FOLDERS, n_cores=DEFAULT_CORES)
