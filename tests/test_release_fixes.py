import importlib.util
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial.distance import pdist

from bactoscoop import utilities as u
from bactoscoop.image import Image
from bactoscoop.imagecollection import ImageCollection
from bactoscoop.signalcorrelation import SignalCorrelation


class TestSplitDaughterBounds(unittest.TestCase):
    def test_both_daughters_must_be_within_bounds(self):
        for sizes, accepted in (
            ((4, 4), True), ((3, 4), False), ((4, 3), False),
            ((801, 4), False), ((4, 801), False),
        ):
            with self.subTest(sizes=sizes):
                image = Image(np.zeros((30, 30)), "field_C1.tif", 0)
                parent = np.column_stack([
                    np.arange(20), np.zeros(20),
                    np.arange(20), np.ones(20),
                ])
                image.mesh_dataframe = pd.DataFrame({
                    "mesh": [parent],
                    "contour": [np.zeros((4, 2))],
                    "midline": [np.zeros((20, 2))],
                })
                daughters = [
                    (np.zeros((size, 4)), np.zeros((4, 2)), np.zeros((size, 2)))
                    for size in sizes
                ]
                with patch.object(image, "get_inverted_image"), patch(
                    "bactoscoop.image.u.interp2d", return_value=None
                ), patch(
                    "bactoscoop.image.u.constr_degree_single_cell_min",
                    return_value=(0.5, None, None, 5, None),
                ), patch(
                    "bactoscoop.image.u.split_point",
                    return_value=(0, 0, np.zeros((1, 4)), np.zeros((1, 4))),
                ), patch(
                    "bactoscoop.image.u.split_mesh2mesh", side_effect=daughters
                ):
                    image.split_cells(thresh=0.35, CD_width=True)
                self.assertEqual(len(image.processed_mesh_dataframe), 2 if accepted else 0)


class TestFiniteSignalCorrelation(unittest.TestCase):
    def test_undefined_ratio_and_infinite_input_are_missing(self):
        self.assertTrue(np.isnan(SignalCorrelation.ratio(1.0, 0.0)))
        self.assertTrue(np.isnan(SignalCorrelation.ratio(np.inf, 2.0)))
        self.assertEqual(SignalCorrelation.ratio(0.0, 2.0), 0.0)
        self.assertFalse(SignalCorrelation.is_valid_signal([1.0, np.inf]))
        self.assertFalse(SignalCorrelation.is_valid_signal(-np.inf))
        self.assertTrue(SignalCorrelation.is_valid_signal([0.0, 2.0]))
        df = pd.DataFrame({"C2_area": [1.0, np.inf], "C3_area": [0.0, 2.0]})
        result = SignalCorrelation(df, "C2", "C3", "area", "ratio").calculate()
        self.assertTrue(result["C2_C3_area_ratio"].isna().all())


class TestFeretMemoryBound(unittest.TestCase):
    def test_matches_full_pairwise_diameter_including_collinear_masks(self):
        rng = np.random.default_rng(20260924)
        masks = [
            rng.random((19, 23)) > 0.76,
            np.eye(13, dtype=bool),
            np.array([[True]]),
            np.array([[True, True]]),
        ]
        for mask in masks:
            with self.subTest(shape=mask.shape):
                coords = np.column_stack(np.where(mask))
                expected = np.max(pdist(coords)) if len(coords) > 1 else 0.0
                self.assertAlmostEqual(u._max_feret_diameter_px(coords), expected)

    def test_large_mask_never_uses_quadratic_pdist(self):
        mask = np.ones((200, 300), dtype=np.uint8)
        with patch("bactoscoop.utilities.spatial.distance.pdist",
                   side_effect=AssertionError("quadratic distance allocation")):
            features = u.get_additional_regionprops_features(mask, 1.0)
        self.assertAlmostEqual(
            features["F_MAX_FERET_DIAMETER"], np.hypot(199, 299)
        )


class TestSubsetSegmentation(unittest.TestCase):
    def test_reordered_subset_is_used_for_evaluation_and_saving(self):
        from bactoscoop.omni import Omnipose

        omni = Omnipose.__new__(Omnipose)
        omni.imgs = [np.full((2, 2), i) for i in range(3)]
        omni.files = [f"image_{i}.tif" for i in range(3)]
        omni.model = MagicMock()
        omni.use_GPU = False
        masks = [np.full((2, 2), i) for i in (2, 0)]
        flows = ["flow_2", "flow_0"]
        omni.model.eval.return_value = (masks, flows, None)
        with patch("bactoscoop.omni.io.save_masks") as save:
            omni.process(n=[2, 0])
            omni.save_masks()
        evaluated = omni.model.eval.call_args.args[0]
        self.assertEqual([int(array[0, 0]) for array in evaluated], [2, 0])
        self.assertEqual([int(array[0, 0]) for array in save.call_args.args[0]], [2, 0])
        self.assertEqual(save.call_args.args[1], masks)
        self.assertEqual(save.call_args.args[2], flows)
        self.assertEqual(save.call_args.args[3], ["image_2.tif", "image_0.tif"])
        with self.assertRaises(ValueError):
            omni.process(n=[])
        with self.assertRaises(ValueError):
            omni.process(n=[0, 0])
        with self.assertRaises(RuntimeError):
            omni.save_masks()


class TestCheckpointFormats(unittest.TestCase):
    def test_failed_pickle_write_preserves_previous_file(self):
        with TemporaryDirectory() as folder:
            collection = ImageCollection(folder)
            collection._to_pickle("checkpoint.pkl", {"version": 1})
            original = (Path(folder) / "checkpoint.pkl").read_bytes()

            def interrupted_dump(data, handle, protocol=None):
                handle.write(b"partial")
                raise OSError("interrupted")

            with patch("bactoscoop.imagecollection.pickle.dump",
                       side_effect=interrupted_dump):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    collection._to_pickle("checkpoint.pkl", {"version": 2})
            self.assertEqual((Path(folder) / "checkpoint.pkl").read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [Path(folder) / "checkpoint.pkl"])
            with (Path(folder) / "checkpoint.pkl").open("rb") as handle:
                self.assertEqual(pickle.load(handle), {"version": 1})

            with patch("bactoscoop.imagecollection.os.replace",
                       side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    collection._to_pickle("checkpoint.pkl", {"version": 3})
            self.assertEqual((Path(folder) / "checkpoint.pkl").read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [Path(folder) / "checkpoint.pkl"])

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow unavailable")
    def test_mesh_and_feature_parquet_roundtrip_in_image_folder(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "masks").mkdir()
            tifffile.imwrite(root / "field_C1.tif", np.zeros((12, 12), dtype=np.uint8))
            tifffile.imwrite(root / "masks" / "field_C1_cp_masks.tif",
                             np.zeros((12, 12), dtype=np.uint8))
            contour = np.array([[2., 2.], [2., 8.], [8., 8.], [8., 2.]])
            mesh = np.array([[2., 2., 2., 8.], [4., 2., 4., 8.],
                             [6., 2., 6., 8.], [8., 2., 8., 8.]])
            midline = (mesh[:, :2] + mesh[:, 2:]) / 2
            collection = ImageCollection(folder)
            collection.mesh_df_collection = pd.DataFrame({
                "image_name": ["field_C1.tif"], "frame": [0], "cell_id": [7],
                "contour": [contour], "mesh": [mesh], "midline": [midline],
            })
            collection.meshdata_to_parquet("curated_meshdata.parquet")
            self.assertTrue((root / "curated_meshdata.parquet").is_file())
            loaded = ImageCollection(folder)
            loaded.batch_load_mesh("curated_meshdata.parquet", phase_channel="C1")
            self.assertEqual(len(loaded.image_objects[0].cells), 1)
            np.testing.assert_array_equal(loaded.image_objects[0].cells[0].mesh, mesh)
            np.testing.assert_array_equal(loaded.image_objects[0].cells[0].contour, contour)
            np.testing.assert_array_equal(loaded.image_objects[0].cells[0].midline, midline)

            collection.merged_features = pd.DataFrame({
                "image_name": ["field_C1.tif"], "cell_id": [7],
                "profile": [[1.0, 2.0, 3.0]],
            })
            collection.dataframe_to_parquet()
            features = pd.read_parquet(root / f"{root.name}_features.parquet")
            self.assertEqual(features["cell_id"].iloc[0], 7)
            self.assertEqual(list(features["profile"].iloc[0]), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
