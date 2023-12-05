"""Functions for fine-tuning and running the RESNET34 model for bird vs forest recognition

Followed this guide with some modifications:
https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data/notebook
"""
from pathlib import Path
from time import sleep
from itertools import islice
import requests
from duckduckgo_search import DDGS
from fastcore.foundation import L

from fastai.vision.utils import download_images, verify_images, resize_images
from fastai.data.transforms import get_image_files
from requests.exceptions import SSLError


def search_images(term, max_images=30):
    """ Search term and return list of urls from duckduckgo with an optional max amount.

    author mango: https://www.kaggle.com/mrmangoes
    :param term: String term to search.
    :param max_images: Maximum number of images to search (default is 30)
    :return: List of string urls of images of the term
    """
    ddgs = DDGS()
    print(f"searching for '{term}'")
    keywords = term
    ddgs_images = ddgs.images(keywords)
    limited_images = list(islice(ddgs_images, max_images))
    return L(limited_images).itemgot('image')


def delete_failed_images(images_path):
    """ Delete failed images

    :param images_path: Path object containing the potentially failed images are.
    :return: Number of images failed, return -1 for exception
    """
    try:
        failed = verify_images(get_image_files(images_path))
        failed.map(Path.unlink, missing_ok=True)
        num_failed = len(failed)
        print(f'Failed images: {num_failed}')
        return num_failed
    except PermissionError:
        return -1


def is_url_image(image_url) -> bool:
    """
    Check header of url for content-type image.

    :param image_url: String, url.
    :return: Boolean, if the url has content-type image.
    """
    image_formats = ("image/png", "image/jpeg", "image/jpg", "image/jpg!d")
    try:
        return requests.head(image_url).headers.get("content-type") in image_formats
    except:
        return False


def download_images_for_categories(category_paths, subjects=None, max_size=400):
    """
    Download images from DuckDuckGo for the specified categories and subjects to the specified paths.

    :param category_paths: Dictionary where keys are category names and values are Path objects.
    :param subjects: List of subjects to search for.
    :param max_size: Maximum image size (default is 400).
    """

    def download(primary, secondary=""):
        """
        :param primary: Primary search string, placed in front.
        :param secondary: (Optional) Secondary search string, placed in back.
        """
        found_urls = search_images(f'{primary}{"" if len(secondary) else " "}{secondary}')
        image_urls = [url for url in found_urls if is_url_image(url)]
        print(f'Downloading {len(image_urls)} images.')
        download_images(category_path, urls=image_urls)
        sleep(10)

    if subjects is None:
        subjects = []
    for category, category_path in category_paths.items():
        if (len(subjects)) > 0:
            for subject in subjects:
                download(category, subject)
        else:
            download(category)
        resize_images(category_path, max_size=max_size, dest=category_path)
        delete_failed_images(category_path)


def create_category_directories(categories, path):
    """ Create directories for each category in the specified base path.

    :param categories: A list of category names (strings).
    :param path: Path object containing the base directory where the category directories will be created.
    :return: Dictionary containing the categories as keys and paths as values, return none for exception
    """
    try:
        categories_paths_dict = {}
        for category in categories:
            category_path = path / category
            category_path.mkdir(exist_ok=True, parents=True)
            categories_paths_dict[category] = category_path
        return categories_paths_dict
    except PermissionError:
        return None


def path_contains_images(path):
    """ Check if given directory contains image files at that directory level (.jpg, .png, .jpeg, .gif, .bmp)

    :param path: Path object containing the directory of the images
    :return: True if any images in path
    """
    try:
        return any(file.suffix.lower() in {'.jpg', '.png', '.jpeg', '.gif', '.bmp'} for file in path.rglob('*'))
    except FileNotFoundError:
        return False


def is_images_setup(paths):
    """ Checks paths to see if the images are already setup

    :param paths: List of Path objects containing the directory of the category to check for images
    :return: True if every directory has images false if not
    """
    return all(path_contains_images(path) for path in paths)
