"""Module containing tests for checking if the path given contains images"""
import shutil
import unittest
from pathlib import Path
from project.computer_vision.setup_utils import path_contains_images


class TestPathContainsImages(unittest.TestCase):
    def setUp(self):
        """Setup function"""
        self.test_base_directory = Path('test_path_contains_test_directory')

    def tearDown(self):
        """Remove test image directories and their files. """
        shutil.rmtree(self.test_base_directory, ignore_errors=True)

    def test_nonexistent_directory(self):
        """Test when the given directory doesn't exist"""
        test_path = self.test_base_directory / "nonexistent/directory"
        self.assertFalse(path_contains_images(test_path))

    def test_directory_is_file(self):
        """Test when the directory given is a file and not a directory"""
        test_path_base = self.test_base_directory / "directory_as_file"
        test_path_base.mkdir(parents=True, exist_ok=True)
        test_path = (test_path_base / 'not_a_dir.txt')
        test_path.touch()
        self.assertFalse(path_contains_images(test_path))

    def test_non_image_file(self):
        """Test when the directory given contains a non image file"""
        test_path = self.test_base_directory / 'non_image_file'
        test_path.mkdir(parents=True, exist_ok=True)
        (test_path / "test.txt").touch()
        self.assertFalse(path_contains_images(test_path))

    def test_no_images(self):
        """Test when directory has no image files"""
        test_path = self.test_base_directory / "empty"
        self.assertFalse(path_contains_images(test_path))

    def test_jpg_image(self):
        """Test when directory contains a .jpg file"""
        test_path = self.test_base_directory / "jpg"

        test_path.mkdir(parents=True, exist_ok=True)
        (test_path / "test.jpg").touch()
        self.assertTrue(path_contains_images(test_path))

    def test_png_image(self):
        """Test when directory contains a .png file"""
        test_path = self.test_base_directory / "png"
        test_path.mkdir(parents=True, exist_ok=True)
        (test_path / "test.png").touch()
        self.assertTrue(path_contains_images(test_path))

    def test_multiple_image_types(self):
        """Test when directory contains multiple image types"""
        test_path = self.test_base_directory / 'multi_images'
        test_path.mkdir(parents=True, exist_ok=True)
        (test_path / "test1.jpg").touch()
        (test_path / "test2.png").touch()
        (test_path / "test3.gif").touch()
        self.assertTrue(path_contains_images(test_path))


if __name__ == '__main__':
    unittest.main()
