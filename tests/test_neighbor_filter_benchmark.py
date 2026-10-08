import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from bactoscoop.imagecollection import ImageCollection
from tests.benchmark_neighbor_prefilter import summarize_prefilter_stats
import bactoscoop.utilities as u


class TestNeighborPrefilterBenchmarkHelpers(unittest.TestCase):
    def test_diagonal_touch_only_counts_in_eight_connectivity(self):
        labels = np.array(
            [
                [1, 0],
                [0, 2],
            ],
            dtype=np.uint16,
        )

        counts_4 = u.get_label_neighbor_counts(labels, connectivity=1)
        counts_8 = u.get_label_neighbor_counts(labels, connectivity=8)

        self.assertEqual(counts_4[1], 0)
        self.assertEqual(counts_4[2], 0)
        self.assertEqual(counts_8[1], 1)
        self.assertEqual(counts_8[2], 1)

    def test_filter_stats_account_for_all_labels(self):
        labels = np.array(
            [
                [0, 2, 0],
                [4, 1, 3],
                [0, 5, 0],
            ],
            dtype=np.uint16,
        )

        filtered_mask, stats = u.filter_labels_by_neighbor_count(
            labels,
            max_neighbors=3,
            connectivity=1,
        )

        self.assertEqual(stats["total_label_count"], 5)
        self.assertEqual(
            stats["kept_label_count"] + stats["removed_label_count"],
            stats["total_label_count"],
        )
        self.assertEqual(filtered_mask.dtype, np.int32)

    def test_summarize_prefilter_stats_normalizes_removed_labels(self):
        dataframe = pd.DataFrame(
            [
                {
                    "image_name": "img",
                    "removed_labels": np.array([1, 2, 3], dtype=np.int32),
                    "removed_label_count": 3,
                }
            ]
        )

        summary = summarize_prefilter_stats(dataframe)

        self.assertEqual(summary[0]["removed_labels"], [1, 2, 3])


class TestNeighborPrefilterBatchProcessIntegration(unittest.TestCase):
    def test_batch_process_mesh_collects_prefilter_stats(self):
        ic = ImageCollection(None)

        first = MagicMock()
        first.image_name = "img_a"
        first.processed_mesh_dataframe = pd.DataFrame({"cell_id": [1]})
        first.prefilter_summary = {
            "removed_label_count": 2,
            "kept_label_count": 4,
            "total_label_count": 6,
            "removed_labels": [5, 6],
        }

        second = MagicMock()
        second.image_name = "img_b"
        second.processed_mesh_dataframe = pd.DataFrame({"cell_id": [2, 3]})
        second.prefilter_summary = {
            "removed_label_count": 0,
            "kept_label_count": 3,
            "total_label_count": 3,
            "removed_labels": [],
        }

        ic.batch_process_mesh(
            object_list=[first, second],
            save_data=False,
            phase_channel="C1",
            join_thresh=4,
            split_thresh=0.5,
            CD_width=False,
            smoothing=0.1,
            neighbor_filter_max_neighbors=3,
            neighbor_filter_connectivity=1,
        )

        first.join_split_pipeline.assert_called_once_with(
            4,
            0.5,
            False,
            0.1,
            neighbor_filter_max_neighbors=3,
            neighbor_filter_connectivity=1,
        )
        second.join_split_pipeline.assert_called_once_with(
            4,
            0.5,
            False,
            0.1,
            neighbor_filter_max_neighbors=3,
            neighbor_filter_connectivity=1,
        )
        self.assertEqual(len(ic.mesh_prefilter_stats), 2)
        self.assertListEqual(
            ic.mesh_prefilter_stats["image_name"].tolist(),
            ["img_a", "img_b"],
        )
        self.assertEqual(len(ic.mesh_df_collection), 3)


if __name__ == "__main__":
    unittest.main()
