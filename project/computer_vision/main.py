"""Module containing driver function and methods for fastai sandbox"""
import logging
import platform
import random
import shutil
import sys
import traceback
from pathlib import Path

import fastai.vision as vision
import torch
from fastai.vision.all import (
    CategoryBlock,
    ClassificationInterpretation,
    DataBlock,
    ImageBlock,
    ImageDataLoaders,
    PILImage,
    RandomResizedCrop,
    RandomSplitter,
    Resize,
    URLs,
    aug_transforms,
    error_rate,
    get_image_files,
    parent_label,
    resnet34,
    untar_data,
    vision_learner,
)
from matplotlib import pyplot
from PIL import Image
from setup_utils import (
    create_category_directories,
    download_images_for_categories,
    is_images_setup,
)
from torchvision.models import resnet18


def try_random_image(learn, test_set_path):
    """
    Randomly selects an image from the test set, predicts its label, displays the image.

    :param learn: Fastai Learner object.
    :param test_set_path: Path object of the directory containing the test set images.
    :return: Tuple with label, label_index, probabilities list.
    """
    image_paths = list(test_set_path.glob("*"))
    if not image_paths:
        print("No images found in the specified directory.")
        return

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
    """Finetune resnet18 for bird vs forest labels.

    :param models_path: Path object for models directory to save fine-tuned model.
    :return: Fastai Learner object.
    """
    model_path = models_path / "bird_vs_forest1.pkl"

    images_path = Path("./images")

    categories = ["bird", "forest"]
    subjects = ["photo", "sun photo", "shade photo"]
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
        item_tfms=[Resize(192, method="squish")],
        batch_tfms=aug_transforms(size=192, min_scale=0.75),
    ).dataloaders(
        images_path, bs=32, num_workers=0
    )  # not sure what to set this to right now. needs to be 0 on windows

    learn = vision.vision_learner(
        dls, resnet18, metrics=error_rate, model_dir=model_path
    )
    if not model_path.is_file():
        # Show batch before training
        # dls.train.show_batch(max_n=4, nrows=1, unique=True)
        # pyplot.show()
        # dls.show_batch(max_n=6)
        learn.fine_tune(3)
        learn.export(model_path)
    return learn


def cat_vs_dog_label_func(animal):
    """Return label of the prediction.

    :param animal: Fastai predict output of the image given.
    :return: label in uppercase
    """
    return animal[0].upper()


def cat_vs_dog_model(models_path):
    """Finetune the resnet32 model for cats vs dog labels

    :return: Fastai Learner object
    """
    model_path = models_path / "cat_vs_dog1.pkl"

    path = untar_data(URLs.PETS) / "images"
    dls = ImageDataLoaders.from_name_func(
        models_path,
        get_image_files(path),
        valid_pct=0.2,
        seed=42,
        label_func=cat_vs_dog_label_func,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224, min_scale=0.75),
        num_workers=0,
    )

    learn = vision_learner(dls, resnet34, metrics=error_rate, model_dir=model_path)
    if not model_path.is_file():
        # Show batch before training
        # dls.train.show_batch(max_n=4, nrows=1, unique=True)
        # pyplot.show()
        learn.fine_tune(1)
        learn.export(model_path)
    return learn


def bear_model_random_resized_crop(models_path):
    """Finetune the resnet32 model for types of bears, grizzly, black, teddy labels

    :return: Fastai Learner object
    """

    model_path = models_path / "bear1.pkl"
    images_path = Path("./images/bear")
    images_path.mkdir(exist_ok=True, parents=True)
    categories = ["grizzly bear", "black bear", "teddy bear"]
    category_paths = create_category_directories(categories, images_path)

    if is_images_setup(category_paths.values()):
        print("images already downloaded")
    else:
        print("downloading images from duckduckgo")
        # noinspection PyBroadException
        try:
            download_images_for_categories(category_paths)
        except Exception:
            print("Something failed.")
            shutil.rmtree(images_path)
            logging.error(traceback.format_exc())
            sys.exit(1)

    bears = DataBlock(
        blocks=[ImageBlock, CategoryBlock],
        get_items=get_image_files,
        splitter=RandomSplitter(seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192)],
        batch_tfms=aug_transforms(size=192, min_scale=0.75),
    )
    # bears = bears.new(item_tfms=[RandomResizedCrop(128, min_scale=0.3)])
    # bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
    # bears = bears.new(item_tfms=Resize(128, ResizeMethod.pad, pad_mode='zeros'))
    # bears = bears.new(item_tfms=[Resize(128)], batch_tfms=aug_transforms(mult=2))

    # Default crops image to square
    bears = bears.new(
        item_tfms=[RandomResizedCrop(128, min_scale=0.3)],
        batch_tfms=aug_transforms(mult=2),
    )
    dls = bears.dataloaders(images_path, num_workers=0)
    learn = vision_learner(dls, resnet34, metrics=error_rate, model_dir=model_path)

    if not model_path.is_file():
        # Show batch before training
        # dls.train.show_batch(max_n=4, nrows=1, unique=True)
        # pyplot.show()
        learn.fine_tune(4)
        learn.export(model_path)

    return learn


def interp_experimentation(model):
    """
    Random things pertaining to using ClassificationInterpretation on a model.
    """
    interp = ClassificationInterpretation.from_learner(model)
    # interp.plot_confusion_matrix()
    interp.plot_top_losses(5, nrows=1)
    pyplot.show()


def main():
    """Driver function

    :return: 0
    """
    os_name = platform.system()
    print(f"NVIDIA GPU available: {torch.cuda.is_available()}")
    print(f"Current cuda device: {torch.cuda.current_device()}")
    print(f"Current OS: {os_name}")

    models_path = Path("./models")
    bear_model = bear_model_random_resized_crop(models_path)
    # try_random_image(bear_model, Path('./images/bear/teddy bear'))

    return 0


if __name__ == "__main__":
    main()
