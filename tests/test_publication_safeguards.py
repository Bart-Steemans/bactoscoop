import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from bactoscoop import utilities as u
from bactoscoop.curation import Curation
from bactoscoop.image import Image
from bactoscoop.imagecollection import ImageCollection


class TestObjectDetectionReplacement(unittest.TestCase):
    def test_one_mesh_exception_keeps_every_contour_and_matching_placeholders(self):
        contours = [
            [[1, 1], [1, 3], [3, 3], [3, 1]],
            [[5, 5], [5, 7], [7, 7], [7, 5]],
        ]
        mesh = np.ones((4, 4), dtype=float)
        midline = np.ones((4, 2), dtype=float)
        with patch.object(
            u, "get_subcellular_objects",
            return_value=(contours, None, np.zeros((10, 10)), 0, 0),
        ), patch.object(u, "draw_mask", return_value=np.zeros((10, 10))), patch.object(
            u, "contour2mesh",
            side_effect=[(contours[0], mesh, midline), IndexError("bad second mesh")],
        ):
            result = u.get_object_mesh(
                np.asarray(contours[0]), np.zeros((10, 10)), None,
                1, 0.1, 3, 4, 0.01, 0.1,
            )

        self.assertEqual(len(result["object_contour"]), 2)
        self.assertEqual(len(result["object_mesh"]), 2)
        self.assertEqual(len(result["object_midline"]), 2)
        self.assertEqual(result["object_mesh"][0].shape, (4, 4))
        self.assertEqual(result["object_mesh"][1].size, 0)
        self.assertEqual(result["object_midline"][1].size, 0)
        self.assertEqual(result["mesh_construction_errors"][0]["object_index"], 1)

    def test_failed_redetection_clears_only_requested_channel(self):
        image = Image(np.zeros((10, 10)), "field_C1.tif", 0)
        image.channels = {"C2": np.zeros((10, 10))}
        cell = SimpleNamespace(
            cell_id=1, contour=np.zeros((4, 2)), mesh=np.zeros((4, 4)),
            object_meshdata={"C2": {"object_contour": ["old"]}, "C3": {"old": True}},
        )
        image.cells = [cell]
        with patch("bactoscoop.image.u.get_object_mesh", side_effect=IndexError("failed")):
            result = image.object_detection(["C2"])

        self.assertTrue(result.empty)
        self.assertNotIn("C2", cell.object_meshdata)
        self.assertIn("C3", cell.object_meshdata)
        self.assertEqual(image.object_detection_errors[0]["error_type"], "IndexError")

    def test_recovered_mesh_failure_does_not_mark_every_detection_failed(self):
        collection = ImageCollection("example")
        collection.phase_channel = "C1"
        image = Image(np.zeros((10, 10)), "field_C1.tif", 0)
        image.cells = [SimpleNamespace(
            cell_id=1, contour=np.zeros((4, 2)), mesh=np.zeros((4, 4)),
            object_meshdata={},
        )]
        collection.image_objects = [image]
        collection.channel_images_by_field = {"C2": {"field": np.zeros((10, 10))}}
        meshdata = {
            "object_contour": [np.zeros((4, 2))],
            "object_mesh": [np.array([])],
            "object_midline": [np.array([])],
            "mesh_construction_errors": [
                {"object_index": 0, "error_type": "IndexError", "error": "bad mesh"}
            ],
        }
        with patch("bactoscoop.image.u.get_object_mesh", return_value=meshdata):
            result = collection.batch_detect_objects(["C2"], reset_channels=False)

        self.assertEqual(len(result), 1)
        self.assertEqual(collection.processing_summary()["cell_error_count"], 1)
        self.assertEqual(collection.processing_summary()["error_count"], 0)


class TestFeatureTableInvalidation(unittest.TestCase):
    def test_successful_incremental_recalculation_overwrites_same_pair(self):
        collection = ImageCollection("example")
        old = pd.DataFrame({"cell_id": [1], "value": [10]})
        new = pd.DataFrame({"cell_id": [1], "value": [20]})
        collection.feature_dataframes = {"objects_C2_features": old}
        collection.image_objects = [SimpleNamespace(
            image_name="field", channels={"C2": np.zeros((2, 2))},
            feature_errors=[], calculate_features=MagicMock(return_value=new),
        )]

        collection.batch_calculate_features([(["C2"], "objects")], reset=False)

        pd.testing.assert_frame_equal(
            collection.feature_dataframes["objects_C2_features"], new
        )

    def test_failed_incremental_recalculation_removes_previous_pair(self):
        collection = ImageCollection("example")
        old = pd.DataFrame({"image_name": ["old"], "cell_id": [1]})
        collection.feature_dataframes = {
            "objects_C2_features": old,
            "profiling_C3_features": old,
        }
        collection.merged_features = old
        collection.image_objects = [SimpleNamespace(
            image_name="new", channels=None, calculate_features=MagicMock(),
        )]

        collection.batch_calculate_features([(["C2"], "objects")], reset=False)

        self.assertNotIn("objects_C2_features", collection.feature_dataframes)
        self.assertIn("profiling_C3_features", collection.feature_dataframes)
        self.assertIsNone(collection.merged_features)

    def test_recreating_images_invalidates_previous_feature_tables(self):
        collection = ImageCollection("example")
        collection.images = [np.zeros((2, 2))]
        collection.image_filenames = ["field_C1.tif"]
        collection.masks = [np.zeros((2, 2))]
        collection.mask_filenames = ["field_C1_cp_masks.tif"]
        collection.feature_dataframes = {"objects_C2_features": pd.DataFrame({"a": [1]})}
        collection.merged_features = pd.DataFrame({"a": [1]})
        with patch.object(
            collection, "create_image_object", return_value=SimpleNamespace()
        ):
            collection.create_image_objects(phase_channel="C1")

        self.assertEqual(collection.feature_dataframes, {})
        self.assertIsNone(collection.merged_features)


class TestCurationStateAndSchema(unittest.TestCase):
    def test_no_save_still_updates_in_memory_meshes(self):
        collection = ImageCollection("example")
        cells = [SimpleNamespace(
            cell_id=cell_id, contour=np.zeros((4, 2)),
            mesh=np.zeros((4, 4)), midline=np.zeros((4, 2)),
        ) for cell_id in (1, 2)]
        collection.image_objects = [SimpleNamespace(
            image_name="field_C1.tif", frame=0, cells=cells,
        )]
        collection.mesh_df_collection = pd.DataFrame({"cell_id": [1, 2]})
        labels = pd.DataFrame({
            "image_name": ["field_C1.tif", "field_C1.tif"],
            "frame": [0, 0], "cell_id": [1, 2], "label": [1, 0],
        })

        def calculate(*_args, **_kwargs):
            collection.feature_dataframes = {"svm_None_features": labels.copy()}

        with patch.object(collection, "batch_calculate_features", side_effect=calculate), patch.object(
            Curation, "compiled_curation", return_value=labels
        ), patch.object(Curation, "get_label_proportions", return_value=(0.5, 0.5)), patch.object(
            collection, "_to_pickle"
        ) as save:
            collection.curate_dataset("unused", save_curated_data=False)

        save.assert_not_called()
        self.assertEqual([cell.cell_id for cell in cells], [1])
        self.assertEqual(collection.mesh_df_collection["cell_id"].tolist(), [1])

    def test_model_schema_is_read_from_model_and_checked_in_order(self):
        data = pd.DataFrame({
            "image_name": ["field"], "frame": [0], "cell_id": [1],
            "b": [2.0], "a": [1.0],
        })
        curation = Curation(data)
        curation.prepare_dataframe(cols=4)
        model = MagicMock()
        model.predict.return_value = np.array([1])

        model.feature_names_in_ = np.array(["a", "b"])
        curation.svm_model = model
        self.assertEqual(curation.make_predictions()["label"].tolist(), [1])

        model.predict.reset_mock()
        model.feature_names_in_ = np.array(["b", "a"])
        with self.assertRaisesRegex(ValueError, "training schema"):
            curation.make_predictions()
        model.predict.assert_not_called()

        del model.feature_names_in_
        with self.assertRaisesRegex(ValueError, "no feature_names_in_"):
            curation.make_predictions()


if __name__ == "__main__":
    unittest.main()
