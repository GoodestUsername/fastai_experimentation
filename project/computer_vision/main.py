"""Module containing driver function and methods for fastai sandbox"""
import random
import platform
import torch

from pathlib import Path
from PIL import Image

from fastcore.all import *
from fastai.vision.all import *

from torchvision.models import resnet18
from setup_utils import create_category_directories, download_images_for_categories, is_images_setup


def test_random_image(learn, test_set_path):
    """
    Randomly selects an image from the test set, predicts its label, and displays the image.

    :param learn: Fastai Learner object.
    :param test_set_path: Path object of the directory containing the test set images.
    :return: Tuple with label, label_index, probabilities list, return -1 if no images found in the directory.
    """
    image_paths = list(test_set_path.glob("*"))
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


def bird_vs_forest_model(models_path):
    """ Finetune resnet18 for bird vs forest labels.

    :param models_path: Path object for models directory to save fine-tuned model.
    :return: Tuple with Fastai Learner object, dictionary of category image paths.
    """
    images_path = Path('./images')

    categories = ['bird', 'forest']
    subjects = ['photo', 'sun photo', 'shade photo']
    category_paths = create_category_directories(categories, images_path)

    if is_images_setup(category_paths.values()):
        print("images already downloaded")
    else:
        print("downloading images from duckduckgo")
        download_images_for_categories(category_paths, subjects)

    dls = DataBlock(
        blocks=[ImageBlock, CategoryBlock],
        get_items=get_image_files,
        splitter=RandomSplitter(seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')],
        batch_tfms=aug_transforms(size=192, min_scale=0.75)
    ).dataloaders(images_path, bs=32, num_workers=0)  # not sure what to set this to right now. needs to be 0 on windows

    dls.show_batch(max_n=6)
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)
    learn.export(models_path / 'bird_vs_forest1.pkl')
    return learn, category_paths


def cat_vs_dog_label_func(animal):
    """ Return label of the prediction.

    :param animal: Fastai predict output of the image given.
    :return: label in uppercase
    """
    return animal[0].upper()


def cat_vs_dog_model(models_path):
    """ Finetune the resnet32 model for cats vs dog labels

    :return: Fastai Learner object
    """

    path = untar_data(URLs.PETS) / 'images'
    dls = ImageDataLoaders.from_name_func(
        models_path,
        get_image_files(path),
        valid_pct=0.2,
        seed=42,
        label_func=cat_vs_dog_label_func,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224, min_scale=0.75),
        num_workers=0
    )

    learn = vision_learner(dls, resnet34, metrics=error_rate, model_dir=models_path)
    learn.fine_tune(1)
    learn.export('cat_vs_dog1.pkl')
    return learn


def main():
    """Driver function

    :return: 0
    """
    os_name = platform.system()
    print(f'NVIDIA GPU available: {torch.cuda.is_available()}')
    print(f'Current cuda device: {torch.cuda.current_device()}')
    print(f'Current OS: {os_name}')

    model_path = Path('./models')
    cats_vs_dogs_model = cat_vs_dog_model(model_path)
    return 0


if __name__ == "__main__":
    main()