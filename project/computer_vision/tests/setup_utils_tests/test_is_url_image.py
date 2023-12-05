"""Module contains tests cases for is_url_image."""
import unittest
from unittest.mock import MagicMock, patch

import requests

from project.computer_vision.setup_utils import is_url_image


class TestIsUrlImage(unittest.TestCase):
    def test_image_url_valid_url_is_image_png(self):
        """Mock the requests head function for png to simulate image url."""
        with patch('requests.head') as mock_head:
            mock_response_image = MagicMock()
            mock_response_image.headers = {"content-type": "image/png"}
            mock_head.return_value = mock_response_image
            self.assertTrue(is_url_image('https://example.com/image.png'))

    def test_image_url_valid_url_is_image_jpeg(self):
        """Mock the requests head function for jpeg to simulate image url."""
        with patch('requests.head') as mock_head:
            mock_response_image = MagicMock()
            mock_response_image.headers = {"content-type": "image/jpeg"}
            mock_head.return_value = mock_response_image
            self.assertTrue(is_url_image('https://example.com/image.jpeg'))

    def test_image_url_valid_url_is_image_jpg_d(self):
        """Mock the requests head function for jpg!d to simulate image url."""
        with patch('requests.head') as mock_head:
            mock_response_image = MagicMock()
            mock_response_image.headers = {"content-type": "image/jpg!d"}
            mock_head.return_value = mock_response_image
            self.assertTrue(is_url_image('https://example.com/image.jpg!d'))

    def test_image_url_valid_url_is_image_jpg(self):
        """Mock the requests head function for jpg to simulate image url."""
        with patch('requests.head') as mock_head:
            mock_response_image = MagicMock()
            mock_response_image.headers = {"content-type": "image/jpg"}
            mock_head.return_value = mock_response_image
            self.assertTrue(is_url_image('https://example.com/image.jpg'))

    def test_image_url_valid_url_not_image(self):
        """Mock the requests head function to simulate image and non-image url."""
        with patch('requests.head') as mock_head:
            mock_response_non_image = MagicMock()
            mock_response_non_image.headers = {"content-type": "text/html"}
            mock_head.return_value = mock_response_non_image
            self.assertFalse(is_url_image('https://example.com/page.html'))

    def test_invalid_url(self):
        """Mock the requests head function to simulate a request error for an invalid URL."""
        with patch('requests.head') as mock_head:
            mock_head.side_effect = requests.RequestException("Invalid URL")
            with self.assertRaises(requests.RequestException):
                is_url_image('invalid_url')


if __name__ == '__main__':
    unittest.main()
