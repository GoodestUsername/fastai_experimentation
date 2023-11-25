"""Module contains tests cases for is_images_setup."""
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from project.computer_vision.setup_utils import is_images_setup


class TestIsImagesSetup(unittest.TestCase):
    def setUp(self):
        """Setup function"""
        self.test_base_directory = Path('test_path_contains_test_directory')

    def tearDown(self):
        """Remove test image directories and their files."""
        shutil.rmtree(self.test_base_directory, ignore_errors=True)

    @patch('project.computer_vision.setup_utils.path_contains_images', return_value=True)
    def test_all_directories_have_images(self, mock_contains_images):
        """Test when all directories contain images"""
        paths = [
            self.test_base_directory / "with/images1",
            self.test_base_directory / "with/images2",
            self.test_base_directory / "with/images3"
        ]
        self.assertTrue(is_images_setup(paths))

    @patch('project.computer_vision.setup_utils.path_contains_images')
    def test_some_directories_missing_images(self, mock_contains_images):
        """Test when some directories contain images and some don't"""
        paths = [
            self.test_base_directory / "with/images1",
            self.test_base_directory / "without/images",
            self.test_base_directory / "with/images3"
        ]
        mock_contains_images.side_effect = [True, False, True]
        self.assertFalse(is_images_setup(paths))

    def test_empty_directory_list(self):
        """Test when an empty list of directories is provided"""
        paths = []
        self.assertTrue(is_images_setup(paths))

    @patch('project.computer_vision.setup_utils.path_contains_images')
    def test_directory_does_not_exist(self, mock_contains_images):
        """Test when one of the directories in the list doesn't exist"""
        paths = [
            self.test_base_directory/ "existing/directory",
            self.test_base_directory / "nonexistent/directory"
        ]
        mock_contains_images.side_effect = [True, False]
        self.assertFalse(is_images_setup(paths))

    @patch('project.computer_vision.setup_utils.path_contains_images')
    def test_directory_is_file(self, mock_contains_images):
        """Test when one of the paths in the list is a file instead of a directory"""
        paths = [
            self.test_base_directory / "directory",
            self.test_base_directory / "file_instead_of_directory.txt"
        ]
        mock_contains_images.side_effect = [True, False]
        self.assertFalse(is_images_setup(paths))

    @patch('project.computer_vision.setup_utils.path_contains_images')
    def test_nested_folders_with_images(self, mock_contains_images):
        """Test when a directory contains nested directories with images"""
        test_path = self.test_base_directory / "very/deeply/nested/folders"
        test_path.mkdir(parents=True, exist_ok=True)
        nested_dir = test_path / "nested"
        nested_dir.mkdir()
        (nested_dir / "image.jpg").touch()

        paths = [test_path]
        mock_contains_images.side_effect = [True]
        self.assertTrue(is_images_setup(paths))


if __name__ == '__main__':
    unittest.main()
