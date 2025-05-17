from __future__ import annotations

import collections
import os.path
import pathlib
import pickle
from collections import defaultdict

import cv2
import imagehash
import keras
import numpy as np
import sklearn.model_selection
import tensorflow as tf
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

from ShogiNeuralNetwork import data_info
from extra.types import ImageNP, Figure, Direction, FilePath


class CellsDataset:
    """
    Class for dataset of cell images
    Stores images of each board cell, its figure and direction
    """

    # Stores hash of images (full images of boards, NOT each cell) used to create dataset
    # Useful for me because I always forget which set of images I used and which not
    __visited_images_hashes: set[imagehash.ImageHash]

    # Dataset fields
    __images: list[ImageNP]  # List of cell images. Could be different sizes
    __figures: list[Figure]
    __directions: list[Direction]

    def __init__(self):
        self.__visited_images_hashes = set()
        self.__images = []
        self.__figures = []
        self.__directions = []

    def is_image_visited(self, img_path: FilePath) -> bool:
        """Returns True if image has already been added to this dataset"""
        img = Image.open(img_path)
        img_hash = imagehash.average_hash(img)
        return img_hash in self.__visited_images_hashes

    def add_image_hash(self, img_path: FilePath) -> None:
        """Adds hash of image to dataset so that it won't permit adding it again"""
        img = Image.open(img_path)
        img_hash = imagehash.average_hash(img)
        self.__visited_images_hashes.add(img_hash)

    def save_yolo(self, path: FilePath) -> None:
        """Saves dataset to YOLO dataset format in two variants: figure and direction"""
        data = list(zip(self.__images, self.__figures, self.__directions))
        train, valid = sklearn.model_selection.train_test_split(data)
        train = list(zip(*train))
        valid = list(zip(*valid))

        for (images, figures, directions), dir_name in [
            (train, "train"),
            (valid, "val"),
            (valid, "test"),
        ]:
            count = defaultdict(int)
            for i in tqdm.trange(len(images), desc=f"Saving YOLO dataset ({dir_name})"):
                img = images[i]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # cv2 saves in bgr
                figure = figures[i]
                direction = directions[i]
                count[figure] += 1
                count[direction] += 1
                figure_img_name = f"{count[figure]}.jpg"
                direction_img_name = f"{count[direction]}.jpg"
                figure_dir_path = os.path.join(path, "figure", dir_name, figure.name)
                direction_dir_path = os.path.join(path, "direction", dir_name, direction.name)
                os.makedirs(figure_dir_path, exist_ok=True)
                os.makedirs(direction_dir_path, exist_ok=True)
                cv2.imwrite(os.path.join(figure_dir_path, figure_img_name), img)
                cv2.imwrite(os.path.join(direction_dir_path, direction_img_name), img)

    @classmethod
    def load_pickle(cls, path: FilePath) -> CellsDataset:
        """Loads dataset from pickle"""
        with open(path, "rb") as f:
            pkl_data = pickle.load(f)
            ds = CellsDataset()
            ds.__visited_images_hashes = pkl_data[0]
            ds.__images = pkl_data[1]
            ds.__figures = pkl_data[2]
            ds.__directions = pkl_data[3]
            return ds

    def save_pickle(self, path: FilePath) -> None:
        """Saves dataset to pickle"""
        with open(path, "wb") as f:
            pkl_data = [
                self.__visited_images_hashes,
                self.__images,
                self.__figures,
                self.__directions,
            ]
            pickle.dump(pkl_data, f)

    @classmethod
    def load(cls, path: FilePath) -> CellsDataset:
        """Loads dataset"""
        path = pathlib.Path(path)
        ds = CellsDataset()
        for img_path in tqdm.tqdm(list(path.glob("*/*/*.jpg")), desc="Loading dataset"):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            figure = Figure[img_path.parent.parent.stem]
            direction = Direction[img_path.parent.stem]
            ds.__images.append(img)
            ds.__figures.append(figure)
            ds.__directions.append(direction)

        if (path / "images_hash.pickle").exists():
            with open(path / "images_hash.pickle", "rb") as f:
                ds.__visited_images_hashes = pickle.load(f)

        return ds

    def save(self, path: FilePath) -> None:
        """Saves dataset
        Folders structure: :path/FIGURE/DIRECTION/*.jpg
        """
        count = defaultdict(int)
        for i in tqdm.trange(len(self.__images), desc="Saving dataset"):
            img = self.__images[i]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # cv2 saves in bgr
            figure = self.__figures[i]
            direction = self.__directions[i]
            count[(figure, direction)] += 1
            img_name = f"{count[(figure, direction)]}.jpg"
            dir_path = os.path.join(path, figure.name, direction.name)
            img_path = os.path.join(dir_path, img_name)
            os.makedirs(dir_path, exist_ok=True)
            cv2.imwrite(img_path, img)
        img_hash_path = os.path.join(path, "images_hash.pickle")
        with open(img_hash_path, "wb") as f:
            pickle.dump(self.__visited_images_hashes, f)

    def add_image(self, cell_img: ImageNP, figure: Figure, direction: Direction) -> None:
        """Add cell image to dataset"""
        self.__images.append(cell_img)
        self.__figures.append(figure)
        self.__directions.append(direction)

    def train_test_split(
            self,
            test_fraction: float = 0.2,
            random_state: int = None,
    ) -> tuple[CellsDataset, CellsDataset]:
        train_images, test_images, train_figures, test_figures, train_directions, test_directions = train_test_split(
            self.__images,
            self.__figures,
            self.__directions,
            test_size=test_fraction,
            random_state=random_state,
        )

        train = CellsDataset()
        train.__images = train_images
        train.__figures = train_figures
        train.__directions = train_directions

        test = CellsDataset()
        test.__images = test_images
        test.__figures = test_figures
        test.__directions = test_directions

        return train, test


    def to_tf_dataset(
            self,
            cell_image_size: int,
            batch_size: int = 64,
            augment: bool = True,
            to_grayscale: bool = True,
            shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Converts dataset to tf.data.Dataset pipeline with augmentation, batching, resizing and scaling
        Dataset has two outputs: "figure" and "direction"
        """

        # preprocessing images so that they are same size, float in range [0, 1] and grayscale if needed
        def process(img: np.ndarray):
            if to_grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (cell_image_size, cell_image_size))
            img = np.expand_dims(img, axis=-1)
            img = img.astype("float32") / 255.0
            return img

        images = np.array([
            process(img)
            for img in self.__images
        ])

        # Converting enums to integers
        figures = self.figure_labels()
        directions = self.direction_labels()

        ds = tf.data.Dataset.from_tensor_slices(
            (
                images,
                {"figure": figures, "direction": directions}
            )
        )
        if shuffle:
            ds = ds.shuffle(ds.cardinality())
        ds = ds.batch(batch_size)
        if augment:
            augmentation = keras.Sequential([
                keras.layers.RandomGaussianBlur(),
                keras.layers.RandomRotation(factor=0.05),
                keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                keras.layers.RandomZoom(height_factor=0.05),
                keras.layers.RandomBrightness(factor=0.2, value_range=[0.0, 1.0]),
                keras.layers.RandomErasing(value_range=[0, 1]),
            ])
            ds = ds.map(lambda img, outputs: (augmentation(img), outputs))
        ds = ds.prefetch(5)

        return ds

    def class_counts(self, figure=True, direction=True) -> dict[Figure | Direction | tuple[Figure, Direction], int]:
        if figure and not direction:
            return collections.Counter(self.__figures)
        elif direction and not figure:
            return collections.Counter(self.__directions)
        else:
            return collections.Counter(zip(self.__figures, self.__directions))

    def class_weights(self) -> tuple[dict, dict]:
        """Returns weights for each class so that rare classes weigh more"""
        n = len(self.__images)
        figure_counts = collections.Counter(self.__figures)
        figure_weights = {data_info.FIGURE_TO_INDEX[figure]: n / count for figure, count in figure_counts.items()}
        direction_counts = collections.Counter(self.__directions)
        direction_weights = {data_info.DIRECTION_TO_INDEX[direction]: n / count for direction, count in direction_counts.items()}
        return figure_weights, direction_weights

    def figure_labels(self) -> np.ndarray:
        return np.array([data_info.FIGURE_TO_INDEX[figure] for figure in self.__figures])

    def direction_labels(self) -> np.ndarray:
        return np.array([data_info.DIRECTION_TO_INDEX[direction] for direction in self.__directions])

    def __len__(self) -> int:
        return len(self.__images)

    def __getitem__(self, i: int) -> tuple[ImageNP, Figure, Direction]:
        return self.__images[i], self.__figures[i], self.__directions[i]
