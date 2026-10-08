import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from time import perf_counter


CURATION_DATASET = (
    "C:/Users/Bart/250821_randompoles_drive1_drive2_drive3_keio_collection.pkl"
)
CHANNEL1 = "C1"
CHANNEL2 = "C2"
CHANNEL3 = "C3"
CHANNEL4 = "C4"
CHANNEL5 = "C5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the optional neighbor prefilter on a copied dataset."
    )
    parser.add_argument(
        "--package-root",
        required=True,
        help="Directory containing the bactoscoop package to import.",
    )
    parser.add_argument(
        "--source-dataset",
        required=True,
        help="Source dataset directory to copy before running the benchmark.",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Directory where a temporary dataset copy and outputs will be created.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name for the copied benchmark dataset directory.",
    )
    parser.add_argument(
        "--neighbor-filter-max-neighbors",
        type=int,
        default=None,
        help="Optional max-neighbor threshold to enable the dev prefilter.",
    )
    parser.add_argument(
        "--neighbor-filter-connectivity",
        type=int,
        default=1,
        help="Connectivity mode for the dev prefilter (1/4 or 2/8).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the JSON summary.",
    )
    return parser.parse_args()


def import_bactoscoop(package_root):
    package_root = str(Path(package_root).resolve())
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    import bactoscoop

    return bactoscoop


def copy_dataset(source_dataset, workspace, run_name):
    source = Path(source_dataset).resolve()
    workspace = Path(workspace).resolve()
    dataset_copy = workspace / run_name
    if dataset_copy.exists():
        shutil.rmtree(dataset_copy)
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dataset_copy)
    return dataset_copy


def summarize_prefilter_stats(dataframe):
    if dataframe is None:
        return None
    if getattr(dataframe, "empty", True):
        return []
    records = []
    for record in dataframe.to_dict(orient="records"):
        cleaned = dict(record)
        if "removed_labels" in cleaned and cleaned["removed_labels"] is not None:
            cleaned["removed_labels"] = list(cleaned["removed_labels"])
        records.append(cleaned)
    return records


def run_pipeline(
    bactoscoop,
    dataset_dir,
    neighbor_filter_max_neighbors=None,
    neighbor_filter_connectivity=1,
):
    params = {
        "phase_channel": CHANNEL1,
        "join_thresh": 4,
        "split_thresh": 0.5,
        "CD_width": False,
        "smoothing": 0.1,
        "save_data": True,
    }
    detect_outer = {
        "channels": [CHANNEL2],
        "reset_channels": True,
        "smoothing": 0.1,
        "align": True,
        "log_sigma": 3,
        "kernel_width": 3,
        "min_overlap_ratio": 0.001,
        "max_external_ratio": 0.4,
    }
    detect_other = {
        "channels": [CHANNEL3, CHANNEL4, CHANNEL5],
        "reset_channels": False,
        "smoothing": 0.1,
        "align": False,
        "log_sigma": 3,
        "kernel_width": 3,
        "min_overlap_ratio": 0.001,
        "max_external_ratio": 0.3,
    }

    feature_tuple1 = [
        ([CHANNEL2], "membrane"),
        ([CHANNEL2], "profiling"),
        ([CHANNEL2], "objects"),
    ]
    feature_tuple2 = [
        ([None], "morphological"),
        ([CHANNEL3], "membrane"),
        ([CHANNEL3, CHANNEL4, CHANNEL5], "profiling"),
        ([CHANNEL3, CHANNEL4, CHANNEL5], "objects"),
    ]
    signal_feature_tuples_1 = [
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
    signal_feature_tuples_2 = [
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

    folder_name = Path(dataset_dir).name
    mesh_pkl_name = f"{folder_name}_meshdata.pkl"
    features_pkl_name = f"{folder_name}_features_4.pkl"

    ic = bactoscoop.ImageCollection(str(dataset_dir))

    stage_times = {}

    started = perf_counter()
    ic.create_image_objects(phase_channel=CHANNEL1)
    stage_times["create_image_objects_seconds"] = round(perf_counter() - started, 3)

    mesh_kwargs = dict(params)
    if neighbor_filter_max_neighbors is not None:
        mesh_kwargs["neighbor_filter_max_neighbors"] = neighbor_filter_max_neighbors
        mesh_kwargs["neighbor_filter_connectivity"] = neighbor_filter_connectivity

    started = perf_counter()
    ic.batch_process_mesh(**mesh_kwargs)
    stage_times["batch_process_mesh_seconds"] = round(perf_counter() - started, 3)
    raw_mesh_rows = int(len(ic.mesh_df_collection))

    started = perf_counter()
    ic.batch_load_mesh(mesh_pkl_name, phase_channel=CHANNEL1)
    stage_times["batch_load_mesh_seconds"] = round(perf_counter() - started, 3)

    started = perf_counter()
    ic.curate_dataset(CURATION_DATASET, control=False)
    stage_times["curate_dataset_seconds"] = round(perf_counter() - started, 3)
    curated_mesh_rows = int(len(ic.mesh_df_collection))
    curated_kept_cells = int(sum(len(image.cells) for image in ic.image_objects))

    started = perf_counter()
    outer_detection_df = ic.batch_detect_objects(**detect_outer)
    stage_times["detect_outer_seconds"] = round(perf_counter() - started, 3)

    started = perf_counter()
    other_detection_df = ic.batch_detect_objects(**detect_other)
    stage_times["detect_other_seconds"] = round(perf_counter() - started, 3)

    started = perf_counter()
    ic.batch_calculate_features(
        feature_tuple1,
        all_data=False,
        reset=True,
        shift_signal=True,
        max_mesh_size=1000,
    )
    stage_times["calculate_features_tuple1_seconds"] = round(
        perf_counter() - started, 3
    )

    started = perf_counter()
    ic.batch_calculate_features(
        feature_tuple2,
        all_data=False,
        reset=False,
        shift_signal=False,
        max_mesh_size=1000,
    )
    stage_times["calculate_features_tuple2_seconds"] = round(
        perf_counter() - started, 3
    )

    started = perf_counter()
    ic.merge_dataframes(include_metadata_tag=True, discard_morphological_nan=True)
    stage_times["merge_dataframes_seconds"] = round(perf_counter() - started, 3)

    started = perf_counter()
    ic.batch_calculate_signal_correlation_features(
        ic.merged_features,
        [CHANNEL2, CHANNEL3, CHANNEL4, CHANNEL5],
        feature_method_tuples=signal_feature_tuples_1,
    )
    stage_times["signal_corr_tuple1_seconds"] = round(perf_counter() - started, 3)

    started = perf_counter()
    ic.batch_calculate_signal_correlation_features(
        ic.merged_features,
        [CHANNEL2, CHANNEL3],
        feature_method_tuples=signal_feature_tuples_2,
    )
    stage_times["signal_corr_tuple2_seconds"] = round(perf_counter() - started, 3)

    started = perf_counter()
    ic.dataframe_to_pkl(pkl_name="features_4")
    stage_times["dataframe_to_pkl_seconds"] = round(perf_counter() - started, 3)

    feature_rows = int(len(ic.merged_features))
    outer_detection_rows = int(len(outer_detection_df))
    other_detection_rows = int(len(other_detection_df))

    return {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "raw_mesh_rows": raw_mesh_rows,
        "curated_mesh_rows": curated_mesh_rows,
        "curated_kept_cells": curated_kept_cells,
        "feature_rows": feature_rows,
        "outer_detection_rows": outer_detection_rows,
        "other_detection_rows": other_detection_rows,
        "prefilter_stats": summarize_prefilter_stats(
            getattr(ic, "mesh_prefilter_stats", None)
        ),
        "stage_times_seconds": stage_times,
        "feature_pkl_path": str((Path(dataset_dir) / features_pkl_name).resolve()),
    }


def main():
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("BACTOSCOOP_LOG_LEVEL", "WARNING")

    dataset_copy = copy_dataset(args.source_dataset, args.workspace, args.run_name)
    bactoscoop = import_bactoscoop(args.package_root)

    started = perf_counter()
    summary = run_pipeline(
        bactoscoop,
        dataset_copy,
        neighbor_filter_max_neighbors=args.neighbor_filter_max_neighbors,
        neighbor_filter_connectivity=args.neighbor_filter_connectivity,
    )
    summary["package_root"] = str(Path(args.package_root).resolve())
    summary["source_dataset"] = str(Path(args.source_dataset).resolve())
    summary["neighbor_filter_max_neighbors"] = args.neighbor_filter_max_neighbors
    summary["neighbor_filter_connectivity"] = args.neighbor_filter_connectivity
    summary["total_elapsed_seconds"] = round(perf_counter() - started, 3)

    payload = json.dumps(summary, indent=2)
    print(payload)

    if args.json_out:
        json_path = Path(args.json_out).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
