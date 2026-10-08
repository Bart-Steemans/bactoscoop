import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import tifffile
import cv2 as cv
from skimage.feature import graycoprops
from skimage.measure import shannon_entropy

from bactoscoop.image import Image, Cell
from bactoscoop.imagecollection import ImageCollection
from bactoscoop.signalcorrelation import SignalCorrelation
from bactoscoop import utilities as u
from bactoscoop import plot as plotting


class TestImageCollectionRobustness(unittest.TestCase):
    def test_add_channels_rejects_non_2d_channel_data_and_continues(self):
        collection = ImageCollection("example")
        image_objects = [
            SimpleNamespace(image_name="field1_C1.tiff", image=np.zeros((2, 2)))
        ]
        collection.phase_channel = "C1"
        collection.channel_images_by_field = {
            "C2": {"field1": np.ones((2, 2, 3))}
        }

        with patch.object(collection, "load_channel_images"):
            collection.add_channels(image_objects, ["C2"])

        self.assertIsNone(image_objects[0].channels)
        self.assertIn("must be 2D", collection.processing_errors[0]["error"])

    def test_init_allows_none_image_folder_path(self):
        collection = ImageCollection()

        self.assertIsNone(collection.image_folder_path)
        self.assertIsNone(collection.name)

    def test_tiff_reader_matches_case_insensitive_suffix_and_extension(self):
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "Field_C2.TIF"
            tifffile.imwrite(image_path, np.ones((2, 2), dtype=np.uint16))

            images, filenames = u.read_tiff_folder(temp_dir, suffix="c2")

        self.assertEqual(filenames, ["Field_C2.TIF"])
        self.assertEqual(images[0].shape, (2, 2))

    def test_load_phase_images_requires_image_folder_path(self):
        collection = ImageCollection()

        with self.assertRaisesRegex(ValueError, "image_folder_path"):
            collection.load_phase_images()

    @patch("bactoscoop.imagecollection.Image")
    def test_create_image_object_passes_px_to_image(self, mock_image_cls):
        collection = ImageCollection("example", px=0.123)
        mesh_df = pd.DataFrame({"image_name": ["img"]})

        collection.create_image_object(
            image=np.zeros((2, 2)),
            image_name="img",
            index=7,
            mask=np.ones((2, 2)),
            mesh_df=mesh_df,
            px=0.123,
        )

        mock_image_cls.assert_called_once()
        _, kwargs = mock_image_cls.call_args
        self.assertEqual(kwargs["px"], 0.123)
        mock_image_cls.return_value.create_cell_object.assert_called_once_with(
            verbose=False
        )

    def test_add_channels_matches_by_filename_not_sorted_position(self):
        collection = ImageCollection("example")
        image_objects = [
            SimpleNamespace(image_name="field1_C1.tiff", image=np.zeros((2, 2))),
            SimpleNamespace(image_name="field2_C1.tiff", image=np.zeros((2, 2))),
        ]
        collection.phase_channel = "C1"
        collection.channel_images_by_field = {
            "C2": {
                "field2": np.full((2, 2), 2),
                "field1": np.full((2, 2), 1),
            }
        }

        with patch.object(collection, "load_channel_images") as mock_loader:
            collection.add_channels(image_objects, ["C2"])

        mock_loader.assert_called_once_with(["C2"])
        self.assertEqual(int(image_objects[0].channels["C2"][0, 0]), 1)
        self.assertEqual(int(image_objects[1].channels["C2"][0, 0]), 2)

    def test_add_channels_continues_with_matching_fields_and_records_missing(self):
        collection = ImageCollection("example")
        image_objects = [
            SimpleNamespace(image_name="field1_C1.tiff", image=np.zeros((2, 2))),
            SimpleNamespace(image_name="field2_C1.tiff", image=np.zeros((2, 2))),
        ]
        collection.phase_channel = "C1"
        collection.channel_images_by_field = {
            "C2": {"field1": np.ones((2, 2))}
        }

        with patch.object(collection, "load_channel_images"):
            collection.add_channels(image_objects, ["C2"])

        self.assertIsNotNone(image_objects[0].channels)
        self.assertIsNone(image_objects[1].channels)
        summary = collection.processing_summary()
        self.assertEqual(summary["status"], "completed_with_errors")
        self.assertEqual(summary["errors"][0]["image_name"], "field2_C1.tiff")

    def test_sequential_channel_additions_keep_earlier_channels(self):
        collection = ImageCollection("example")
        collection.phase_channel = "C1"
        image = SimpleNamespace(
            image_name="field1_C1.tif", image=np.zeros((2, 2)),
            channels=None, bg_channels={}, chann_interp2d={},
        )
        collection.channel_images_by_field = {
            "C2": {"field1": np.full((2, 2), 2)}
        }
        with patch.object(collection, "load_channel_images"):
            collection.add_channels([image], ["C2"])
            collection.channel_images_by_field["C3"] = {
                "field1": np.full((2, 2), 3)
            }
            collection.add_channels([image], ["C3"])

        self.assertEqual(set(image.channels), {"C2", "C3"})
        self.assertEqual(int(image.channels["C2"][0, 0]), 2)
        self.assertEqual(int(image.channels["C3"][0, 0]), 3)

    def test_detection_with_reset_false_keeps_prior_channel_on_image(self):
        collection = ImageCollection("example")
        collection.phase_channel = "C1"
        image = SimpleNamespace(
            image_name="field1_C1.tif", image=np.zeros((2, 2)),
            channels=None, object_detection=MagicMock(return_value=pd.DataFrame()),
        )
        collection.image_objects = [image]

        def load_tiff(_folder, suffix=""):
            return [np.ones((2, 2))], [f"field1_{suffix}.tif"]

        with patch("bactoscoop.imagecollection.u.read_tiff_folder", side_effect=load_tiff):
            collection.batch_detect_objects(channels=["C2"], reset_channels=True)
            collection.batch_detect_objects(channels=["C3"], reset_channels=False)

        self.assertEqual(set(image.channels), {"C2", "C3"})
        self.assertEqual(collection.processing_summary()["error_count"], 0)

    def test_detection_reattaches_cached_channel_after_image_reload(self):
        collection = ImageCollection("example")
        collection.phase_channel = "C1"
        image = SimpleNamespace(
            image_name="field1_C1.tif", image=np.zeros((2, 2)),
            channels=None, object_detection=MagicMock(return_value=pd.DataFrame()),
        )
        collection.image_objects = [image]
        collection.channel_images_by_field = {
            "C2": {"field1": np.ones((2, 2))}
        }

        with patch.object(collection, "load_channel_images") as loader:
            collection.batch_detect_objects(channels=["C2"], reset_channels=False)

        loader.assert_not_called()
        self.assertIn("C2", image.channels)
        image.object_detection.assert_called_once()

    def test_extra_channel_field_is_recorded(self):
        collection = ImageCollection("example")
        collection.phase_channel = "C1"
        image = SimpleNamespace(
            image_name="field1_C1.tif", image=np.zeros((2, 2)), channels=None
        )
        collection.channel_images_by_field = {
            "C2": {"field1": np.ones((2, 2)), "field9": np.ones((2, 2))}
        }

        with patch.object(collection, "load_channel_images"):
            collection.add_channels([image], ["C2"])

        self.assertIn("C2", image.channels)
        self.assertEqual(collection.processing_errors[0]["image_name"], "field9")

    def test_internal_object_detection_error_is_recorded(self):
        collection = ImageCollection("example")
        collection.phase_channel = "C1"
        image = Image(np.zeros((2, 2)), "field1_C1.tif", 0)
        image.cells = [SimpleNamespace(
            cell_id=7, contour=np.zeros((3, 2)), mesh=np.zeros((2, 4)),
            object_meshdata={},
        )]
        collection.image_objects = [image]
        collection.channel_images_by_field = {"C2": {"field1": np.ones((2, 2))}}

        with patch("bactoscoop.image.u.get_object_mesh", side_effect=IndexError("bad mesh")):
            collection.batch_detect_objects(channels=["C2"], reset_channels=False)

        summary = collection.processing_summary()
        self.assertEqual(summary["cell_error_count"], 1)
        self.assertEqual(summary["cell_errors"][0]["error_type"], "IndexError")
        self.assertEqual(summary["status"], "completed_with_errors")

    def test_shift_correction_reattaches_cached_channel_and_records_cell_error(self):
        collection = ImageCollection("example")
        collection.phase_channel = "C1"
        image = Image(np.zeros((2, 2)), "field1_C1.tif", 0)
        image.cells = [SimpleNamespace(
            cell_id=7, contour=np.zeros((3, 2)), mesh=np.zeros((2, 4)),
            midline=np.zeros((2, 2)),
        )]
        collection.image_objects = [image]
        collection.channel_images_by_field = {"C2": {"field1": np.ones((2, 2))}}

        with patch("bactoscoop.image.u.shift_contour", side_effect=RuntimeError("bad shift")):
            collection.batch_shift_correction(["C2"], reset_channels=False)

        summary = collection.processing_summary()
        self.assertIn("C2", image.channels)
        self.assertEqual(summary["cell_error_count"], 1)
        self.assertEqual(summary["cell_errors"][0]["stage"], "cell_shift_correction")
        self.assertEqual(summary["status"], "completed_with_errors")

    def test_cell_feature_error_is_recorded_without_retry_status_or_log(self):
        collection = ImageCollection("example")
        image = SimpleNamespace(
            image_name="field1_C1.tif", channels={"C2": np.ones((2, 2))},
            feature_errors=[{"cell_id": 7, "error": "bad cell"}],
            calculate_features=MagicMock(return_value=pd.DataFrame({"cell_id": [7]})),
        )
        collection.image_objects = [image]

        with patch("bactoscoop.imagecollection.bactoscoop_logger.warning") as warning:
            collection.batch_calculate_features([(["C2"], "profiling")])

        warning.assert_not_called()
        summary = collection.processing_summary()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["cell_error_count"], 1)
        self.assertEqual(summary["cell_errors"][0]["cell_id"], 7)

    def test_all_failed_cell_features_mark_output_incomplete(self):
        collection = ImageCollection("example")
        image = SimpleNamespace(
            image_name="field1_C1.tif", channels={"C2": np.ones((2, 2))},
            feature_errors=[{"cell_id": 7, "error": "bad cell"}],
            calculate_features=MagicMock(return_value=pd.DataFrame()),
        )
        collection.image_objects = [image]

        collection.batch_calculate_features([(["C2"], "profiling")])

        summary = collection.processing_summary()
        self.assertEqual(summary["cell_error_count"], 1)
        self.assertEqual(summary["status"], "completed_with_errors")
        self.assertEqual(summary["error_count"], 1)

    def test_cell_error_logging_can_be_enabled_without_changing_records(self):
        collection = ImageCollection("example", log_cell_errors=True)
        with patch("bactoscoop.imagecollection.bactoscoop_logger.warning") as warning:
            collection._record_cell_error(
                "cell_feature_calculation", RuntimeError("bad cell"),
                "field1", "C2", 7
            )
        warning.assert_called_once()
        self.assertEqual(collection.processing_summary()["cell_error_count"], 1)

    def test_missing_image_channel_skips_cell_feature_calculation(self):
        collection = ImageCollection("example")
        image = SimpleNamespace(
            image_name="field1_C1.tif", channels=None,
            calculate_features=MagicMock(),
        )
        collection.image_objects = [image]

        collection.batch_calculate_features([(["C2"], "profiling")])

        image.calculate_features.assert_not_called()
        self.assertEqual(collection.processing_summary()["cell_error_count"], 0)
        self.assertEqual(collection.processing_errors[0]["stage"], "image_feature_calculation")

    def test_strict_error_policy_raises_on_missing_channel_field(self):
        collection = ImageCollection("example", error_policy="raise")
        image_objects = [
            SimpleNamespace(image_name="field2_C1.tiff", image=np.zeros((2, 2)))
        ]
        collection.phase_channel = "C1"
        collection.channel_images_by_field = {"C2": {}}

        with patch.object(collection, "load_channel_images"):
            with self.assertRaisesRegex(ValueError, "No C2 image matches field"):
                collection.add_channels(image_objects, ["C2"])

    def test_create_image_objects_skips_faulty_image_and_keeps_rest(self):
        collection = ImageCollection("example")
        collection.masks = [np.ones((2, 2)), np.ones((2, 2))]
        collection.mask_filenames = ["img1_masks.tif", "img2_masks.tif"]
        collection.images = [np.zeros((2, 2)), np.zeros((2, 2))]
        collection.image_filenames = ["img1", "img2"]

        with patch.object(
            collection,
            "create_image_object",
            side_effect=[MagicMock(image_name="img1"), RuntimeError("bad image")],
        ):
            collection.create_image_objects()

        self.assertEqual(len(collection.image_objects), 1)
        self.assertEqual(collection.image_objects[0].image_name, "img1")

    def test_field_key_normalizes_channel_tag_before_mask_suffix(self):
        self.assertEqual(
            u.image_field_key("Sample_B1_XY16_C1.tiff", suffix="C1"),
            u.image_field_key("Sample_B1_XY16_C1_cp_masks.tif", is_mask=True),
        )

    def test_phase_mask_pairing_rejects_mask_from_wrong_channel(self):
        collection = ImageCollection("example")
        collection.phase_channel = "C1"
        collection.images = [np.zeros((2, 2))]
        collection.image_filenames = ["field1_C1.tif"]
        collection.masks = [np.ones((2, 2))]
        collection.mask_filenames = ["field1_C2_cp_masks.tif"]

        with self.assertRaisesRegex(ValueError, "No phase images have matching masks"):
            collection.create_image_objects()

        self.assertEqual(collection.processing_summary()["error_count"], 2)

    def test_failed_segmentation_prevents_loading_stale_mask_files(self):
        collection = ImageCollection("example")
        collection.images = [np.zeros((2, 2))]
        collection.image_filenames = ["field1_C1.tif"]
        collection.segmentation_failed = True

        with patch.object(collection, "load_masks") as load_masks:
            with self.assertRaisesRegex(RuntimeError, "Segmentation failed"):
                collection.create_image_objects()

        load_masks.assert_not_called()

    def test_phase_mask_pairing_uses_field_identity_when_counts_match(self):
        collection = ImageCollection("example")
        collection.phase_channel = "C1"
        collection.images = [np.full((2, 2), 1), np.full((2, 2), 2)]
        collection.image_filenames = ["field1_C1.tif", "field2_C1.tif"]
        collection.masks = [np.full((2, 2), 10), np.full((2, 2), 30)]
        collection.mask_filenames = ["field1_cp_masks.tif", "field3_cp_masks.tif"]

        received = []
        def capture(image, image_name, index, mask, mesh_df, px):
            received.append((image_name, int(image[0, 0]), int(mask[0, 0]), index))
            return SimpleNamespace(image_name=image_name)

        with patch.object(collection, "create_image_object", side_effect=capture):
            collection.create_image_objects()

        self.assertEqual(received, [("field1_C1.tif", 1, 10, 0)])
        self.assertEqual(collection.processing_summary()["error_count"], 2)

    def test_curation_mapping_removes_rejected_cell_from_matching_image(self):
        collection = ImageCollection("example")
        cell_a = SimpleNamespace(cell_id=1)
        cell_b = SimpleNamespace(cell_id=1)
        collection.image_objects = [
            SimpleNamespace(image_name="field2", frame=2, cells=[cell_a]),
            SimpleNamespace(image_name="field3", frame=3, cells=[cell_b]),
        ]
        collection.curated_df = pd.DataFrame(
            {"image_name": ["field3"], "frame": [3], "cell_id": [1], "label": [0]}
        )

        # Exercise the same mapping/removal path used by curate_dataset without
        # invoking feature calculation or loading an SVM model.
        collection._apply_curation_labels()

        self.assertEqual(collection.image_objects[0].cells, [cell_a])
        self.assertEqual(collection.image_objects[1].cells, [])

    def test_curation_falls_back_to_actual_frame_without_image_name(self):
        collection = ImageCollection("example")
        cell_a = SimpleNamespace(cell_id=1)
        cell_b = SimpleNamespace(cell_id=1)
        collection.image_objects = [
            SimpleNamespace(image_name="field2", frame=2, cells=[cell_a]),
            SimpleNamespace(image_name="field3", frame=3, cells=[cell_b]),
        ]
        collection.curated_df = pd.DataFrame(
            {"frame": [3], "cell_id": [1], "label": [0]}
        )

        collection._apply_curation_labels()

        self.assertEqual(collection.image_objects[0].cells, [cell_a])
        self.assertEqual(collection.image_objects[1].cells, [])

    def test_curation_does_not_remove_cell_when_name_and_frame_disagree(self):
        collection = ImageCollection("example")
        cell = SimpleNamespace(cell_id=1)
        collection.image_objects = [
            SimpleNamespace(image_name="field2", frame=2, cells=[cell])
        ]
        collection.curated_df = pd.DataFrame(
            {"image_name": ["field2"], "frame": [3], "cell_id": [1], "label": [0]}
        )

        collection._apply_curation_labels()

        self.assertEqual(collection.image_objects[0].cells, [cell])
        self.assertEqual(collection.processing_errors[0]["stage"], "curation_mapping")

    def test_control_plot_uses_stored_frame_after_skipped_image(self):
        contour = np.array([[0, 0], [0, 1], [1, 1]], dtype=float)
        image2 = SimpleNamespace(
            frame=2, image=np.full((2, 2), 2),
            cells=[SimpleNamespace(cell_id=1, contour=contour)],
        )
        image3 = SimpleNamespace(
            frame=3, image=np.full((2, 2), 3),
            cells=[SimpleNamespace(cell_id=1, contour=contour)],
        )

        with patch.object(plotting, "plt") as pyplot:
            plotting.plot_svm_controls([(1, 3)], [image2, image3], "Positive")

        self.assertIs(pyplot.imshow.call_args.args[0], image3.image)

    def test_phase_mask_pairing_rejects_non_2d_images(self):
        collection = ImageCollection("example")
        collection.phase_channel = "C1"
        collection.images = [np.zeros((2, 2, 3))]
        collection.image_filenames = ["field1_C1.tif"]
        collection.masks = [np.ones((2, 2, 3))]
        collection.mask_filenames = ["field1_C1_cp_masks.tif"]

        with self.assertRaisesRegex(ValueError, "No phase images have matching masks"):
            collection.create_image_objects()

        self.assertEqual(collection.processing_summary()["error_count"], 1)

    def test_curation_mapping_records_cell_id_not_found(self):
        collection = ImageCollection("example")
        collection.image_objects = [
            SimpleNamespace(image_name="field1", frame=1, cells=[])
        ]
        collection.curated_df = pd.DataFrame(
            {"image_name": ["field1"], "frame": [1], "cell_id": [7], "label": [1]}
        )

        collection._apply_curation_labels()

        self.assertEqual(collection.processing_errors[0]["stage"], "curation_mapping")
        self.assertIn("not found", collection.processing_errors[0]["error"])

    def test_create_image_objects_uses_replaced_mesh_dataframe_after_id_reuse(self):
        collection = ImageCollection("example")
        collection.images = [np.zeros((2, 2))]
        collection.masks = [np.ones((2, 2))]
        collection.mask_filenames = ["img1_masks.tif"]
        collection.image_filenames = ["img1"]
        collection.mesh_df_collection = pd.DataFrame(
            {"image_name": ["img1", "img1"], "cell_id": [1, 2]}
        )
        received_mesh_dataframes = []

        def capture_mesh_dataframe(
            image, image_name, index, mask, mesh_df, px
        ):
            received_mesh_dataframes.append(mesh_df)
            return MagicMock(image_name=image_name)

        # Reuse one synthetic ID for both dataframe instances to reproduce the
        # former cache's collision without relying on allocator behavior.
        with patch("bactoscoop.imagecollection.id", return_value=1, create=True):
            with patch.object(
                collection,
                "create_image_object",
                side_effect=capture_mesh_dataframe,
            ):
                collection.create_image_objects()
                collection.mesh_df_collection = pd.DataFrame(
                    {"image_name": ["img1"], "cell_id": [7]}
                )
                collection.create_image_objects()

        self.assertEqual(received_mesh_dataframes[0]["cell_id"].tolist(), [1, 2])
        self.assertEqual(received_mesh_dataframes[1]["cell_id"].tolist(), [7])

    def test_create_image_objects_refreshes_lookup_after_in_place_mesh_edit(self):
        collection = ImageCollection("example")
        collection.images = [np.zeros((2, 2))]
        collection.masks = [np.ones((2, 2))]
        collection.mask_filenames = ["img1_masks.tif"]
        collection.image_filenames = ["img1"]
        collection.mesh_df_collection = pd.DataFrame(
            {"image_name": ["img1", "img1"], "cell_id": [1, 2]}
        )
        received_mesh_dataframes = []

        def capture_mesh_dataframe(
            image, image_name, index, mask, mesh_df, px
        ):
            received_mesh_dataframes.append(mesh_df)
            return MagicMock(image_name=image_name)

        with patch.object(
            collection,
            "create_image_object",
            side_effect=capture_mesh_dataframe,
        ):
            collection.create_image_objects()
            collection.mesh_df_collection.drop(index=1, inplace=True)
            collection.create_image_objects()

        self.assertEqual(received_mesh_dataframes[0]["cell_id"].tolist(), [1, 2])
        self.assertEqual(received_mesh_dataframes[1]["cell_id"].tolist(), [1])

    def test_merge_dataframes_merges_normal_case(self):
        collection = ImageCollection("example")
        collection.feature_dataframes = {
            "morphological_None_features": pd.DataFrame(
                {
                    "image_name": ["img"],
                    "cell_id": [1],
                    "frame": [0],
                    "cell_area": [2.5],
                    "cell_length": [5.0],
                }
            ),
            "profiling_C2_features": pd.DataFrame(
                {
                    "image_name": ["img"],
                    "cell_id": [1],
                    "frame": [0],
                    "mean_signal": [3.2],
                }
            ),
        }

        merged = collection.merge_dataframes(include_metadata_tag=True)

        self.assertEqual(len(merged), 1)
        self.assertIn("Metadata_image_name", merged.columns)
        self.assertIn("C2_mean_signal", merged.columns)
        self.assertAlmostEqual(merged.loc[0, "cell_area"], 2.5)
        self.assertAlmostEqual(merged.loc[0, "C2_mean_signal"], 3.2)

    def test_merge_dataframes_preserves_C2_C3_and_full_underscore_channel(self):
        collection = ImageCollection("example")
        identity = {"image_name": ["img"], "cell_id": [1], "frame": [0]}
        collection.feature_dataframes = {
            "profiling_C2_features": pd.DataFrame({**identity, "mean_signal": [2.0]}),
            "profiling_C3_features": pd.DataFrame({**identity, "mean_signal": [3.0]}),
            "profiling_GFP_green_features": pd.DataFrame(
                {**identity, "mean_signal": [4.0]}
            ),
        }

        merged = collection.merge_dataframes()

        self.assertEqual(
            set(merged.columns),
            {
                "image_name", "cell_id", "frame", "C2_mean_signal",
                "C3_mean_signal", "GFP_green_mean_signal",
            },
        )
        self.assertEqual(merged.loc[0, "C2_mean_signal"], 2.0)
        self.assertEqual(merged.loc[0, "C3_mean_signal"], 3.0)
        self.assertEqual(merged.loc[0, "GFP_green_mean_signal"], 4.0)

    def test_mask_crop_preserves_foreground_on_rectangular_images(self):
        cases = [
            ((10, 30), slice(2, 5), slice(20, 25), 0, (20, 2), (3, 5)),
            ((30, 10), slice(20, 25), slice(2, 5), 0, (2, 20), (5, 3)),
            ((10, 30), slice(8, 10), slice(28, 30), 2, (26, 6), (4, 4)),
            ((30, 10), slice(28, 30), slice(8, 10), 2, (6, 26), (4, 4)),
        ]
        for shape, rows, columns, pad, expected_offset, expected_shape in cases:
            with self.subTest(shape=shape, rows=rows, columns=columns, pad=pad):
                mask = np.zeros(shape, dtype=np.uint8)
                mask[rows, columns] = 1

                _, cropped, _, x, y = u.crop_image(mask_to_crop=mask, pad=pad)

                self.assertEqual((x, y), expected_offset)
                self.assertEqual(cropped.shape, expected_shape)
                self.assertEqual(int(cropped.sum()), int(mask.sum()))

    def test_sinuosity_keeps_nonclosed_diagonal_midline(self):
        midline = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])

        self.assertEqual(u.sinuosity(midline), 1.414)
        with self.assertRaisesRegex(ValueError, "closed contour"):
            u.sinuosity(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))

    def test_merge_dataframes_rejects_empty_collection(self):
        collection = ImageCollection("example")

        with self.assertRaisesRegex(ValueError, "No feature dataframes"):
            collection.merge_dataframes()

    def test_batch_shift_correction_rejects_empty_collection(self):
        collection = ImageCollection("example")

        with self.assertRaisesRegex(ValueError, "No image objects"):
            collection.batch_shift_correction(shifted_channel=["C2"])

    def test_batch_calculate_features_rejects_empty_collection(self):
        collection = ImageCollection("example")

        with self.assertRaisesRegex(ValueError, "No image objects"):
            collection.batch_calculate_features([([None], "svm")])


class TestImageShiftCorrection(unittest.TestCase):
    def test_shift_correction_returns_zero_without_cells(self):
        image = Image(
            image=np.zeros((4, 4)),
            image_name="img",
            frame=0,
            mask=np.zeros((4, 4)),
        )
        image.channels = {"C2": np.zeros((4, 4))}

        result = image.shift_correction(
            shifted_channel=["C2"],
            log_sigma=1.5,
            kernel_width=3,
            min_overlap_ratio=0.1,
            max_external_ratio=1.0,
            phase_log_sigma=0.5,
            phase_closing_level=2,
            signal_closing_level=12,
            max_shift_correction=15,
        )

        self.assertEqual(result, 0.0)


class TestObjectMeshFeatureFallback(unittest.TestCase):
    @staticmethod
    def _calculate_object_features(
        mesh_widths, all_data=True, retained_channels=None, channel="C2"
    ):
        shape = (100, 100)
        signal = np.tile(np.arange(100, dtype=np.uint8), (100, 1))
        image = Image(signal, "synthetic_C1.tif", 0, mask=np.zeros(shape, dtype=np.uint8))
        image.channels = {channel: signal}
        image.bg_channels[channel] = signal.astype(float)

        cell_contour = cv.ellipse2Poly((40, 40), (35, 9), 0, 0, 360, 5).astype(float)
        x = np.linspace(5, 75, 36)
        cell_mesh = np.column_stack(
            [x, np.full(36, 32), x, np.full(36, 48)]
        )
        cell_midline = np.column_stack([x, np.full(36, 40)])
        cell = Cell(cell_contour, cell_mesh, cell_midline, shape, 7)

        contours = []
        meshes = []
        midlines = []
        for center, half_width in zip((25, 55), mesh_widths):
            contours.append(
                cv.ellipse2Poly((center, 41), (8, 3), 0, 0, 360, 10).astype(float)
            )
            if half_width is None:
                meshes.append(np.array([]))
                midlines.append(np.array([]))
                continue
            obj_x = np.linspace(center - 8, center + 8, 9)
            obj_mesh = np.column_stack(
                [obj_x, np.full(9, 41 - half_width),
                 obj_x, np.full(9, 41 + half_width)]
            )
            meshes.append(obj_mesh)
            midlines.append((obj_mesh[:, :2] + obj_mesh[:, 2:]) / 2)
        cell.object_meshdata[channel] = {
            "object_contour": contours,
            "object_mesh": meshes,
            "object_midline": midlines,
        }
        image.cells = [cell]
        result = image.calculate_features(
            "objects", channel, all_data, False, False, 1000,
            retain_contour_on_object_mesh_failure_channels=retained_channels,
        )
        return image, cell, result

    def test_bad_mesh_in_any_position_preserves_contour_and_position_features(self):
        _, valid_cell, valid = self._calculate_object_features((2.3, 2.3))
        self.assertEqual(len(valid), 1)
        self.assertFalse(pd.isna(valid["cell_total_obj_mesh_length"].iloc[0]))
        midline_energies, _ = u.object_bending_energy(
            valid_cell.object_meshdata["C2"]["object_midline"], 0.065
        )
        np.testing.assert_allclose(
            valid["obj_midline_bending_energies"].iloc[0], midline_energies
        )

        for widths in ((1.5, 2.3), (2.3, 1.5)):
            with self.subTest(widths=widths):
                image, cell, result = self._calculate_object_features(
                    widths, retained_channels=["C2"]
                )
                self.assertEqual(len(result), 1)
                self.assertEqual(image.feature_errors, [])
                self.assertEqual(len(image.cells), 1)
                self.assertEqual(len(cell.object_meshdata["C2"]["object_contour"]), 2)
                self.assertEqual(result["object_number"].iloc[0], 2)
                self.assertGreater(result["cell_total_obj_area"].iloc[0], 0)
                self.assertFalse(pd.isna(result["avg_pole_obj_distance"].iloc[0]))
                self.assertTrue(pd.isna(result["cell_total_obj_mesh_length"].iloc[0]))
                self.assertTrue(pd.isna(result["cell_avg_obj_width"].iloc[0]))
                self.assertTrue(pd.isna(result["cell_avg_midline_sinuosity"].iloc[0]))
                self.assertTrue(pd.isna(result["obj_midline_bending_energies"].iloc[0]))
                np.testing.assert_allclose(
                    result["obj_volumes_approx"].iloc[0],
                    valid["obj_volumes_approx"].iloc[0],
                )
                self.assertEqual(set(result.columns), set(valid.columns))

        _, _, compact_valid = self._calculate_object_features((2.3, 2.3), False)
        _, _, compact_bad = self._calculate_object_features(
            (1.5, 2.3), False, retained_channels=["C2"]
        )
        self.assertEqual(set(compact_bad.columns), set(compact_valid.columns))

    def test_unconstructed_single_object_mesh_keeps_contour_features(self):
        image, cell, result = self._calculate_object_features(
            (None,), retained_channels=["C2"]
        )

        self.assertEqual(image.feature_errors, [])
        self.assertEqual(len(image.cells), 1)
        self.assertEqual(result["object_number"].iloc[0], 1)
        self.assertGreater(result["cell_total_obj_area"].iloc[0], 0)
        self.assertTrue(pd.isna(result["cell_total_obj_mesh_length"].iloc[0]))
        self.assertEqual(len(cell.object_meshdata["C2"]["object_contour"]), 1)

    def test_closed_object_midline_keeps_contour_features(self):
        image, cell, _ = self._calculate_object_features((2.3,))
        midline = cell.object_meshdata["C2"]["object_midline"][0]
        midline[-1] = midline[0]

        result = image.calculate_features(
            "objects", "C2", False, False, False, 1000,
            retain_contour_on_object_mesh_failure_channels=["C2"],
        )

        self.assertEqual(image.feature_errors, [])
        self.assertEqual(result["object_number"].iloc[0], 1)
        self.assertTrue(pd.isna(result["cell_avg_midline_sinuosity"].iloc[0]))

    def test_default_rejects_bad_mesh_without_deleting_cell_or_detection(self):
        for all_data in (False, True):
            for widths in ((1.5, 2.3), (2.3, 1.5), (None,)):
                with self.subTest(widths=widths, all_data=all_data):
                    image, cell, result = self._calculate_object_features(
                        widths, all_data=all_data
                    )
                    self.assertTrue(result.empty)
                    self.assertEqual(image.feature_errors, [])
                    self.assertEqual(len(image.cells), 1)
                    self.assertEqual(len(cell.object_meshdata["C2"]["object_contour"]), len(widths))
                    self.assertEqual(cell.objects_features, {})

    def test_only_listed_channel_retains_features_on_bad_mesh(self):
        _, _, rejected = self._calculate_object_features(
            (1.5,), retained_channels=["C4"], channel="C2"
        )
        _, _, retained = self._calculate_object_features(
            (1.5,), retained_channels=["C4"], channel="C4"
        )
        self.assertTrue(rejected.empty)
        self.assertEqual(retained["object_number"].iloc[0], 1)
        self.assertTrue(pd.isna(retained["cell_total_obj_mesh_length"].iloc[0]))

    def test_other_methods_are_unaffected_after_objects_rejection(self):
        image, cell, rejected = self._calculate_object_features((1.5,))
        self.assertTrue(rejected.empty)
        for method, channel in (
            ("morphological", None), ("profiling", "C2"), ("membrane", "C2")
        ):
            with self.subTest(method=method):
                with patch(f"bactoscoop.features.Features.{method}") as calculation:
                    calculation.side_effect = lambda c, *args: setattr(
                        c, f"{method}_features", {"marker": 7}
                    )
                    result = image.calculate_features(
                        method, channel, False, False, False, 1000,
                        retain_contour_on_object_mesh_failure_channels=["C2"],
                    )
                self.assertEqual(result["marker"].iloc[0], 7)
                self.assertEqual(len(image.cells), 1)
                self.assertFalse(cell.discard)

    def test_collection_passes_channel_list_and_rejects_string(self):
        collection = ImageCollection("example")
        image = SimpleNamespace(
            image_name="field_C1.tif", channels={"C2": np.ones((2, 2))},
            feature_errors=[],
            calculate_features=MagicMock(
                return_value=pd.DataFrame({"cell_id": [7]})
            ),
        )
        collection.image_objects = [image]
        collection.batch_calculate_features(
            [(["C2"], "objects")],
            retain_contour_on_object_mesh_failure_channels=["C4", "C5"],
        )
        self.assertEqual(image.calculate_features.call_args.args[-1], frozenset({"C4", "C5"}))
        with self.assertRaises(TypeError):
            collection.batch_calculate_features(
                [(["C2"], "objects")],
                retain_contour_on_object_mesh_failure_channels="C4",
            )


class TestCellRecoveryRegressions(unittest.TestCase):
    @patch(
        "bactoscoop.utilities.maybe_reverse_orientation",
        side_effect=lambda contour, result, midline: (contour, result, midline),
    )
    @patch("bactoscoop.utilities.get_object_contours")
    @patch("bactoscoop.utilities.contour2mesh")
    @patch("bactoscoop.utilities.crop_image")
    def test_get_cellular_mesh_processes_existing_labels_only(
        self,
        mock_crop_image,
        mock_contour2mesh,
        mock_get_object_contours,
        _mock_orientation,
    ):
        masks = np.array([[0, 1, 0], [0, 3, 0]], dtype=np.uint8)
        contour = np.array([[0, 0], [1, 0], [1, 1]])
        mesh = np.array([[0, 0, 1, 1]])
        midline = np.array([[0, 0], [1, 1]])

        mock_crop_image.side_effect = (
            lambda mask_to_crop=None, **kwargs: (None, mask_to_crop, None, 0, 0)
        )
        mock_get_object_contours.return_value = [np.array([[0, 0], [1, 1]])]
        mock_contour2mesh.return_value = (contour.copy(), mesh.copy(), midline.copy())

        dataframe = u.get_cellular_mesh(masks, smoothing=0.1)

        self.assertEqual(mock_contour2mesh.call_count, 2)
        self.assertEqual(len(dataframe), 2)


class TestNeighborPrefilter(unittest.TestCase):
    @staticmethod
    def _crowded_mask():
        return np.array(
            [
                [0, 0, 2, 0, 0],
                [0, 0, 2, 0, 0],
                [5, 5, 1, 3, 3],
                [0, 0, 4, 0, 0],
                [0, 0, 4, 0, 0],
            ],
            dtype=np.uint16,
        )

    def test_get_label_neighbor_counts_counts_edge_touching_neighbors(self):
        neighbor_counts = u.get_label_neighbor_counts(
            self._crowded_mask(), connectivity=1
        )

        self.assertEqual(neighbor_counts[1], 4)
        self.assertEqual(neighbor_counts[2], 1)
        self.assertEqual(neighbor_counts[3], 1)
        self.assertEqual(neighbor_counts[4], 1)
        self.assertEqual(neighbor_counts[5], 1)

    def test_filter_labels_by_neighbor_count_removes_crowded_cells_and_relabels(self):
        filtered_mask, stats = u.filter_labels_by_neighbor_count(
            self._crowded_mask(),
            max_neighbors=3,
            connectivity=1,
        )

        self.assertEqual(stats["total_label_count"], 5)
        self.assertEqual(stats["removed_label_count"], 1)
        self.assertEqual(stats["kept_label_count"], 4)
        self.assertEqual(stats["removed_labels"], [1])
        self.assertSetEqual(set(np.unique(filtered_mask)), {0, 1, 2, 3, 4})

    def test_apply_neighbor_prefilter_sets_processing_mask_and_summary(self):
        image = Image(
            image=np.zeros((5, 5)),
            image_name="img",
            frame=0,
            mask=self._crowded_mask(),
        )

        summary = image.apply_neighbor_prefilter(max_neighbors=3, connectivity=1)

        self.assertIsNotNone(summary)
        self.assertIsNotNone(image.processing_mask)
        self.assertEqual(summary["removed_label_count"], 1)
        self.assertEqual(summary["removed_labels"], [1])
        self.assertSetEqual(set(np.unique(image.processing_mask)), {0, 1, 2, 3, 4})


class TestSignalCorrelationRegressions(unittest.TestCase):
    def test_distance_correlation_uses_centered_distances(self):
        x = [0, 1, 2, 3]
        y = [0, 1, 4, 9]

        result = SignalCorrelation.distance_correlation(x, y)

        self.assertAlmostEqual(result, 0.9684641640, places=9)
        self.assertAlmostEqual(
            SignalCorrelation.distance_correlation(np.array(x) / 10, np.array(y) / 10),
            result,
            places=12,
        )
        self.assertAlmostEqual(
            SignalCorrelation.distance_correlation(x, [2, 4, 6, 8]), 1.0
        )
        self.assertTrue(np.isnan(SignalCorrelation.distance_correlation([1, 1, 1], [0, 1, 2])))

    def test_histogram_intersection_uses_shared_normalized_bins(self):
        self.assertEqual(
            SignalCorrelation.histogram_intersection([0, 1], [100, 101]), 0.0
        )
        self.assertEqual(
            SignalCorrelation.histogram_intersection([0, 0, 1, 1], [0, 1]), 1.0
        )
        self.assertEqual(
            SignalCorrelation.histogram_intersection([0, 1], [0, 0, 1, 1]), 1.0
        )
        self.assertEqual(
            SignalCorrelation.histogram_intersection([0, 1], [0, 1], bins=[0, 0.5]),
            1.0,
        )

    @staticmethod
    def _signal_correlation_tuples():
        first_feature_method_tuples = [
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
        second_feature_method_tuples = [
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
        return first_feature_method_tuples, second_feature_method_tuples

    def test_signal_correlation_subset_matches_reference_output(self):
        fixture_path = (
            Path(__file__).resolve().parent / "data" / "signal_correlation_subset.pkl"
        )
        payload = pd.read_pickle(fixture_path)
        reference_df = payload["dataframe"]
        input_df = reference_df[payload["input_columns"]].copy()

        collection = ImageCollection("example")
        first_tuples, second_tuples = self._signal_correlation_tuples()

        actual_df = collection.batch_calculate_signal_correlation_features(
            input_df,
            ["C2", "C3", "C4", "C5"],
            feature_method_tuples=first_tuples,
        )
        actual_df = collection.batch_calculate_signal_correlation_features(
            actual_df,
            ["C2", "C3"],
            feature_method_tuples=second_tuples,
        )

        for column in payload["expected_columns"]:
            # This fixture records the historical metric definitions. The
            # corrected metrics have independent reference tests above.
            if column.endswith(("distance_correlation", "histogram_intersection")):
                continue
            np.testing.assert_allclose(
                actual_df[column].to_numpy(dtype=np.float64),
                reference_df[column].to_numpy(dtype=np.float64),
                rtol=0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_signal_validity_matches_legacy_behavior(self):
        def legacy_is_valid_signal(signal):
            if isinstance(signal, (list, np.ndarray)):
                return all(pd.notna(val) for val in signal) and len(signal) > 0
            return pd.notna(signal)

        samples = [
            [],
            [1.0, 2.0],
            [1.0, np.nan],
            np.array([1.0, 2.0]),
            np.array([1.0, np.nan]),
            5.0,
            np.nan,
        ]

        for sample in samples:
            self.assertEqual(
                SignalCorrelation.is_valid_signal(sample),
                legacy_is_valid_signal(sample),
            )


class TestCuratedMaskExport(unittest.TestCase):
    def test_rectangular_mask_keeps_row_column_contour_in_place(self):
        original_mask = np.zeros((10, 30), dtype=np.uint16)
        original_mask[2:5, 20:25] = 7
        cell = SimpleNamespace(
            cell_id=42,
            contour=np.array([[2, 20], [2, 24], [4, 24], [4, 20]]),
        )

        with TemporaryDirectory() as folder:
            collection = ImageCollection(folder)
            collection.image_objects = [
                SimpleNamespace(
                    image_name="field_C1.tif",
                    frame=0,
                    mask=original_mask,
                    cells=[cell],
                )
            ]

            paths, provenance = collection.export_curated_masks()
            exported_mask = tifffile.imread(paths[0])

        np.testing.assert_array_equal(exported_mask == 1, original_mask == 7)
        self.assertEqual(len(provenance), 1)
        self.assertEqual(provenance.iloc[0]["area_px"], 15)
        self.assertEqual(provenance.iloc[0]["original_mask_labels"], "7")
        self.assertEqual(provenance.iloc[0]["centroid_x_px"], 22)
        self.assertEqual(provenance.iloc[0]["centroid_y_px"], 3)


class TestHaralickHelper(unittest.TestCase):
    def test_glcm_feature_helper_matches_graycoprops(self):
        image = np.array(
            [[0, 1, 2, 2], [1, 2, 3, 3], [2, 3, 4, 5], [5, 5, 6, 7]],
            dtype=np.uint16,
        )
        glcm = u.create_co_occurrence_matrix(
            image,
            distances=[2],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )
        expected = (
            graycoprops(glcm, "dissimilarity").mean(),
            graycoprops(glcm, "correlation").mean(),
            graycoprops(glcm, "homogeneity").mean(),
            graycoprops(glcm, "energy").mean(),
            graycoprops(glcm, "contrast").mean(),
            shannon_entropy(glcm, base=2),
        )

        actual = u.get_glcm_feature_means(glcm)

        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
