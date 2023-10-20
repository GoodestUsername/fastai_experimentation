"""Functions for fine-tuning and running the RESNET34 model for bird vs forest recognition

Followed this guide with some modifications:
https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data/notebook
"""
import random
import glob

from pathlib import Path
from time import sleep

from itertools import islice
from PIL import Image
from duckduckgo_search import DDGS
from fastcore.foundation import L

from fastai.vision.core import PILImage
from fastai.vision.utils import download_images, verify_images, resize_images
from fastai.data.transforms import get_image_files

import torch


def search_images(term, max_images=30):
    """ Search term and return list of urls from duckduckgo with an optional max amount.

    :author mango: https://www.kaggle.com/mrmangoes
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
        failed.map(Path.unlink)
        num_failed = len(failed)
        print(f'Failed images: {num_failed}')
        return num_failed
    except PermissionError:
        return -1


def create_category_directories(categories, path):
    """ Create directories for each category in the specified base path.

    :param categories: A list of category names (strings).
    :param path: Path object containing the base directory where the category directories will be created.
    :return: True for success, return none for exception
    """
    try:
        for category in categories:
            category_path = path / 'images' / category
            category_path.mkdir(exist_ok=True, parents=True)
        return True
    except PermissionError:
        return None


def download_images_for_categories(category_paths, subjects, max_size=400):
    """
    Download images from DuckDuckGo for the specified categories and subjects to the specified paths.

    :param category_paths: Dictionary where keys are category names and values are Path objects.
    :param subjects: List of subjects to search for.
    :param max_size: Maximum image size (default is 400).
    """
    for category, category_path in category_paths.items():
        for subject in subjects:
            found_urls = search_images(f'{category} {subject}')
            download_images(category_path, urls=found_urls)
            sleep(10)
        resize_images(category_path, max_size=max_size, dest=category_path)


def test_random_image(learn, testset_path: Path):
    """
    Randomly selects an image from the test set, predicts its label, and displays the image.

    :param learn: Fastai Learner object.
    :param testset_path: Path object of the directory containing the test set images.
    :return: Tuple with label, label_index, probabilities list, return -1 if no images found in the directory.
    """
    image_paths = list(testset_path.glob("*"))
    if not image_paths:
        print("No images found in the specified directory.")
        return -1

    random_image_path = random.choice(image_paths)
    print(f"File name: {random_image_path}.")

    pil_image = PILImage.create(random_image_path)
    label, label_index, probabilities = learn.predict(pil_image)

    print(f"This is a: {label}.")
    print(f"Probabilities: {probabilities}")
    print(f"Label index: {label_index}")

    im = Image.open(random_image_path)
    im.show()

    return label, label_index, probabilities
