"""Module contains tests for search_images"""
import unittest
from unittest.mock import patch
from project.computer_vision.setup_utils import search_images  # Replace 'your_module' with the actual module name
from fastcore.foundation import L


class TestSearchImages(unittest.TestCase):
    @patch('duckduckgo_search.DDGS.images')
    def test_search_images_with_results(self, mock_images):
        """Test for ddg images returning 3 images"""
        mock_images.return_value = [
            {'image': 'url1'},
            {'image': 'url2'},
            {'image': 'url3'}
        ]

        result = search_images('cats', max_images=3)
        expected_result = L(['url1', 'url2', 'url3'])
        self.assertEqual(result, expected_result)

    @patch('duckduckgo_search.DDGS.images')
    def test_search_images_no_results(self, mock_images):
        """Test for ddg images returning no images"""
        mock_images.return_value = []

        result = search_images('random_term', max_images=5)
        expected_result = L([])
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
