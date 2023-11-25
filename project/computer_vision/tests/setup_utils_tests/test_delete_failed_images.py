"""Module contains tests for delete_failed_images"""
import os.path
import unittest
from pathlib import Path
from unittest.mock import patch

from project.computer_vision.setup_utils import delete_failed_images
import shutil
import glob


class TestDeleteFailedImages(unittest.TestCase):
    sample_good_image_path = Path('good_images')

    def setUp(self):
        """Setup test image directories."""
        self.test_dir_no_images = Path("test_no_images")
        self.test_dir_no_images.mkdir(parents=True, exist_ok=True)

        self.test_dir_good_images = Path("test_good_images")
        self.test_dir_good_images.mkdir(parents=True, exist_ok=True)

        self.test_dir_bad_images = Path("test_bad_images")
        self.test_dir_bad_images.mkdir(parents=True, exist_ok=True)

        self.test_dir_good_and_bad_images = Path("test_both_good_and_bad_images")
        self.test_dir_good_and_bad_images.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Remove test image directories and their files. """
        shutil.rmtree(self.test_dir_no_images, ignore_errors=True)
        shutil.rmtree(self.test_dir_good_images, ignore_errors=True)
        shutil.rmtree(self.test_dir_bad_images, ignore_errors=True)
        shutil.rmtree(self.test_dir_good_and_bad_images, ignore_errors=True)

    @patch('pathlib.Path.unlink')
    def test_delete_failed_images_permission_error(self, mock_unlink):
        """Test for no images in folder"""
        self.test_bad_images = [
            self.test_dir_bad_images / "image1.jpg",
            self.test_dir_bad_images / "image2.jpg",
            self.test_dir_bad_images / "image3.jpg"
        ]
        for file in self.test_bad_images:
            file.touch()
        mock_unlink.side_effect = PermissionError('Unit test delete image perm error.')
        result = delete_failed_images(self.test_dir_bad_images)
        self.assertEqual(-1, result)

    def test_delete_failed_images_no_images(self):
        """Test for no images in folder"""
        result = delete_failed_images(self.test_dir_no_images)
        self.assertEqual(0, result)

    def test_delete_failed_images_with_only_good_images(self):
        """Test for all good images in folder"""
        for good_image in glob.iglob(os.path.join(TestDeleteFailedImages.sample_good_image_path, '*.jpg')):
            shutil.copy(good_image, self.test_dir_good_images)
        result = delete_failed_images(self.test_dir_good_images)
        self.assertEqual(0, result)

    def test_delete_failed_images_with_only_bad_images(self):
        """Test for all bad images in folder"""
        self.test_bad_images = [
            self.test_dir_bad_images / "image1.jpg",
            self.test_dir_bad_images / "image2.jpg",
            self.test_dir_bad_images / "image3.jpg"
        ]
        for file in self.test_bad_images:
            file.touch()
        result = delete_failed_images(self.test_dir_bad_images)
        self.assertEqual(3, result)

    def test_delete_failed_images_with_both_good_and_bad_images(self):
        """Test for both good and bad images in folder"""
        self.test_bad_images = [
            self.test_dir_good_and_bad_images / "image1.jpg",
            self.test_dir_good_and_bad_images / "image2.jpg",
            self.test_dir_good_and_bad_images / "image3.jpg"
        ]

        for file in self.test_bad_images:
            file.touch()
        for good_image in glob.iglob(os.path.join(TestDeleteFailedImages.sample_good_image_path, '*.jpg')):
            shutil.copy(good_image, self.test_dir_good_and_bad_images)

        result = delete_failed_images(self.test_dir_good_and_bad_images)
        self.assertEqual(3, result)


if __name__ == '__main__':
    unittest.main()
