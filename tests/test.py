# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:39:13 2024

@author: Bart Steemans. Govers Lab.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

# Import the module and classes you want to test
from bactoscoop import ImageCollection  # replace with your actual class and module name

class TestYourClassName(unittest.TestCase):
    def setUp(self):
        """
        This method is called before every test. You can use it to set up any objects or state that you'll need in your tests.
        """
        # Create a test instance of your class
        self.test_instance = ImageCollection()
        
        # Set up mock data
        self.test_instance.image_folder_path = "/test/path"
        self.test_instance.mesh_df_collection = pd.DataFrame({'image_name': ['image1', 'image2']})
        self.test_instance.px = 0.5  # Example pixel size

        # Mock image and mask data
        self.test_instance.images = ['image1_data', 'image2_data']
        self.test_instance.masks = ['mask1_data', 'mask2_data']
        self.test_instance.image_filenames = ['image1', 'image2']

    @patch('your_module.u.read_tiff_folder')
    def test_load_masks(self, mock_read_tiff_folder):
        """
        Test the load_masks method to ensure masks are correctly loaded from the '/masks' folder.
        """
        mock_read_tiff_folder.return_value = (['mask1', 'mask2'], ['mask_filename1', 'mask_filename2'])
        
        self.test_instance.load_masks()
        
        self.assertEqual(self.test_instance.masks, ['mask1', 'mask2'])
        self.assertEqual(self.test_instance.mask_filenames, ['mask_filename1', 'mask_filename2'])
    
    @patch('your_module.u.read_tiff_folder')
    def test_load_phase_images(self, mock_read_tiff_folder):
        """
        Test the load_phase_images method to ensure phase images are correctly loaded.
        """
        mock_read_tiff_folder.return_value = (['image1', 'image2'], ['image1_name', 'image2_name'], ['/path1', '/path2'])
        
        self.test_instance.load_phase_images(phase_channel='c1')
        
        self.assertEqual(self.test_instance.images, ['image1', 'image2'])
        self.assertEqual(self.test_instance.image_filenames, ['image1_name', 'image2_name'])
        self.assertEqual(self.test_instance.paths, ['/path1', '/path2'])

    def test_create_image_objects(self):
        """
        Test create_image_objects method to ensure objects are created correctly.
        """
        # Mock the get_mesh_dataframe and create_image_object methods
        self.test_instance.get_mesh_dataframe = MagicMock(return_value=pd.DataFrame({'x': [1, 2], 'y': [3, 4]}))
        self.test_instance.create_image_object = MagicMock(return_value='mock_image_object')

        self.test_instance.create_image_objects()

        # Check that image objects were created
        self.assertEqual(len(self.test_instance.image_objects), 2)
        self.test_instance.create_image_object.assert_called_with(
            'image1_data', 'image1', 0, 'mask1_data', self.test_instance.get_mesh_dataframe('image1'), self.test_instance.px
        )

    def test_add_channels(self):
        """
        Test add_channels method to ensure channels are correctly added to image objects.
        """
        # Mock image objects and channel images
        mock_img_obj_1 = MagicMock()
        mock_img_obj_2 = MagicMock()
        img_objects = [mock_img_obj_1, mock_img_obj_2]

        self.test_instance.channel_images = {'c1': ['c1_image1', 'c1_image2']}
        self.test_instance.add_channels(img_objects, ['c1'])

        # Verify that channels were added to the image objects
        self.assertEqual(mock_img_obj_1.channels, {'c1': 'c1_image1'})
        self.assertEqual(mock_img_obj_2.channels, {'c1': 'c1_image2'})

    def test_get_mesh_dataframe(self):
        """
        Test get_mesh_dataframe method to ensure the correct dataframe is returned.
        """
        # Test with existing mesh data
        result = self.test_instance.get_mesh_dataframe('image1')
        self.assertFalse(result.empty)

        # Test with non-existent mesh data
        result = self.test_instance.get_mesh_dataframe('non_existent_image')
        self.assertIsNone(result)

    def test_create_image_object(self):
        """
        Test create_image_object method to ensure Image objects are created properly.
        """
        mock_image = 'mock_image_data'
        mock_image_name = 'mock_image_name'
        mock_index = 0
        mock_mask = 'mock_mask_data'
        mock_mesh_df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        mock_px = 0.5

        # Assuming Image is a class defined in your module
        with patch('your_module.Image') as MockImage:
            mock_image_instance = MockImage.return_value
            img_obj = self.test_instance.create_image_object(mock_image, mock_image_name, mock_index, mock_mask, mock_mesh_df, mock_px)
            
            MockImage.assert_called_with(mock_image, mock_image_name, mock_index, mock_mask, mock_mesh_df)
            mock_image_instance.create_cell_object.assert_called_once_with(verbose=False)
            self.assertEqual(img_obj, mock_image_instance)

if __name__ == '__main__':
    unittest.main()
