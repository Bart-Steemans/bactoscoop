import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from bactoscoop import utilities as u
from bactoscoop.image import Image
from bactoscoop.imagecollection import ImageCollection


def daughter_coordinates(length):
    x = np.linspace(0, length, 100)
    y = 2 * np.sin(np.linspace(0, np.pi, 100))
    return x, y, x, -y


class TestDaughterMeshLimit(unittest.TestCase):
    def test_extreme_length_is_rejected_before_reconstruction(self):
        with patch.object(u, "mesh2contour") as contour_builder, patch.object(u, "get_avg_width_no_px") as width, patch.object(
            u, "extend_skeleton"
        ) as extend, patch.object(u, "straighten_by_orthogonal_lines") as straighten:
            mesh, contour, midline = u.split_mesh2mesh(*daughter_coordinates(500000))
        self.assertEqual(mesh.shape, (0, 4))
        self.assertEqual(contour.shape, (0, 2))
        self.assertEqual(midline.shape, (0, 2))
        contour_builder.assert_not_called()
        width.assert_not_called()
        extend.assert_not_called()
        straighten.assert_not_called()

    def test_nonfinite_length_is_rejected_before_reconstruction(self):
        for length in (np.nan, np.inf):
            with self.subTest(length=length), np.errstate(invalid="ignore"), patch.object(
                u, "mesh2contour"
            ) as contour_builder, patch.object(u, "extend_skeleton") as extend:
                result = u.split_mesh2mesh(*daughter_coordinates(length))
                self.assertEqual(result[0].shape, (0, 4))
                contour_builder.assert_not_called()
                extend.assert_not_called()

    def test_real_rebuilding_at_custom_limit_and_rounding_boundary(self):
        for length, predicted in ((450, 900), (500, 1000), (500.25, 1000), (500.5, 1001)):
            with self.subTest(length=length):
                args = daughter_coordinates(length)
                self.assertEqual(len(u.split_mesh2mesh(*args)[0]), 0)
                result = u.split_mesh2mesh(*args, max_daughter_cell_mesh_rows=1000)
                self.assertEqual(len(result[0]), predicted if predicted <= 1000 else 0)

    def test_minimum_is_not_applied_before_multiple_poles_add_rows(self):
        left = np.array([[0., 0.], [1., 0.], [2., 0.]])
        right = left + [0., 1.]
        poles = (np.array([[0., 0.], [.1, 0.]]), np.array([[2., 0.]]))
        with patch.object(u, "extend_skeleton", return_value=(left, *poles)), patch.object(
            u, "straighten_by_orthogonal_lines", return_value=(left, right, np.empty(0), left)
        ) as straighten:
            result = u.split_mesh2mesh(*daughter_coordinates(1.5))
        straighten.assert_called_once()
        self.assertEqual(len(result[0]), 4)

    def test_invalid_parameter_raises_at_every_entry_point(self):
        image = Image(np.zeros((10, 10)), "field_c1.tif", 0)
        image.get_inverted_image = MagicMock()
        image.join_cells = MagicMock()
        collection = ImageCollection(None)
        for value in (None, True, np.bool_(True), 3, 0, -1, 800.0, "1000", np.inf):
            for function in (
                lambda: u.split_mesh2mesh(*daughter_coordinates(10), max_daughter_cell_mesh_rows=value),
                lambda: image.split_cells(max_daughter_cell_mesh_rows=value),
                lambda: image.join_split_pipeline(max_daughter_cell_mesh_rows=value),
                lambda: collection.batch_process_mesh(object_list=[], save_data=False,
                    max_daughter_cell_mesh_rows=value),
            ):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "integer >= 4"):
                    function()
        image.get_inverted_image.assert_not_called()
        image.join_cells.assert_not_called()
        self.assertEqual(u._validate_max_daughter_cell_mesh_rows(np.int64(1000)), 1000)

    def test_custom_limit_reaches_split_cells_through_batch_and_pipeline(self):
        image = Image(np.zeros((10, 10)), "field_c1.tif", 0)
        image.join_cells = MagicMock()
        image.mask2mesh = MagicMock()
        image.split_cells = MagicMock()
        image.create_cell_object = MagicMock()
        image.processed_mesh_dataframe = pd.DataFrame(columns=["mesh", "contour", "midline"])
        collection = ImageCollection(None)
        collection.batch_process_mesh(
            object_list=[image], save_data=False, CD_width=True,
            max_daughter_cell_mesh_rows=1000,
        )
        image.split_cells.assert_called_once_with(
            0.35, True, max_daughter_cell_mesh_rows=1000,
        )


if __name__ == "__main__":
    unittest.main()
