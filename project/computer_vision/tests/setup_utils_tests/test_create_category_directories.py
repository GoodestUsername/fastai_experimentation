"""Module contains tests for setting up category directories"""
import unittest
from pathlib import Path
from unittest.mock import patch
from project.computer_vision.setup_utils import create_category_directories


class TestCreateCategoryDirectories(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_base_path = Path("test_base")
        self.test_base_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Remove the temporary directory and its contents after each test
        for child in self.test_base_path.iterdir():
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        self.test_base_path.rmdir()

    @patch('pathlib.Path.mkdir')
    def test_create_category_directories_permission_error(self, mock_mkdir):
        """Test for permission error for making directories"""
        restricted_path = Path("/restricted/path")
        mock_mkdir.side_effect = PermissionError('unit test')
        result = create_category_directories(["category"], restricted_path)

        self.assertIsNone(None, result)

    def test_create_category_directories_no_categories(self):
        """Test the function when no categories are provided"""
        result = create_category_directories([], self.test_base_path)

        self.assertEqual(result, {})

    def test_create_category_directories(self):
        """Test the function with a list of categories"""
        categories = ["category1", "category2", "category3"]
        result = create_category_directories(categories, self.test_base_path)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(categories))

        for category in categories:
            category_path = self.test_base_path / category
            self.assertTrue(category_path.exists())
            self.assertTrue(category_path.is_dir())
            self.assertEqual(result[category], category_path)

    def test_create_category_directories_existing_directories(self):
        """Test case where the directories already exist"""
        existing_categories = ["existing1", "existing2"]
        for category in existing_categories:
            category_path = self.test_base_path / category
            category_path.mkdir(exist_ok=True)

        result = create_category_directories(existing_categories, self.test_base_path)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(existing_categories))

        for category in existing_categories:
            category_path = self.test_base_path / category
            self.assertTrue(category_path.exists())
            self.assertTrue(category_path.is_dir())
            self.assertEqual(result[category], category_path)


if __name__ == '__main__':
    unittest.main()
