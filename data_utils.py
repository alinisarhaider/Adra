from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from matplotlib.image import imread
from sklearn.model_selection import train_test_split


def normalize_data(x: np.ndarray) -> np.ndarray:
    """ This function normalizes the image pixel values from 0 - 255 to 0 - 1. """
    x = x.astype('float32')
    x /= 255.0
    return x


def split_data(x: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                 np.ndarray, np.ndarray]:
    """ This function splits the data into three parts: Train, Validation & Test. """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=seed, shuffle=True,
                                                        stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.6, random_state=seed, stratify=y_test)
    return x_train, x_val, x_test, y_train, y_val, y_test


def augment_data(x: np.ndarray, y: np.ndarray, flipud: bool = True, fliplr: bool = True) -> Tuple[np.ndarray,
                                                                                                  np.ndarray]:
    """ This function augments data by flipping images vertically and horizontally. """
    augmented_x = list()
    augmented_y = list()
    for img, label in zip(x, y):
        ud = np.flipud(img)
        lr = np.fliplr(img)
        udlr = np.fliplr(ud)
        if flipud and fliplr:
            augmented_x.extend(np.stack((img, ud, lr, udlr)))
            augmented_y.extend(np.stack((label, label, label, label)))
        elif flipud:
            augmented_x.extend(np.stack((img, ud)))
            augmented_y.extend(np.stack((label, label)))
        elif fliplr:
            augmented_x.extend(np.stack((img, lr)))
            augmented_y.extend(np.stack((label, label)))
    return np.array(augmented_x), np.array(augmented_y)


def data_reader(global_vars: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                      np.ndarray]:
    """ This function reads data and returns ready to use data for training the model. """
    x = list()
    y = list()
    label = 0
    for class_path in [global_vars['CLASS1_PATH'], global_vars['CLASS2_PATH']]:
        images_dir = Path(class_path)
        for image in sorted(images_dir.glob('*.jpg')):
            img = imread(image)
            if img.shape[0] != 230:
                continue
            x.append(img)
            y.append(label)
        label = 1

    x = np.array(x)
    y = np.array(y)

    # Normalize data
    if global_vars['NORMALIZE_PIXEL_VALUES']:
        x = normalize_data(x)

    # Split data into train, validation and test set
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x=x, y=y, seed=global_vars['SEED'])

    # Augment data
    if global_vars['AUGMENT_TRAIN']:
        x_train, y_train = augment_data(x=x_train, y=y_train, flipud=global_vars['FLIP_UPSIDE_DOWN'],
                                        fliplr=global_vars['FLIP_LEFT_TO_RIGHT'])
    if global_vars['AUGMENT_VAL']:
        x_val, y_val = augment_data(x=x_val, y=y_val, flipud=global_vars['FLIP_UPSIDE_DOWN'],
                                    fliplr=global_vars['FLIP_LEFT_TO_RIGHT'])
    if global_vars['AUGMENT_TEST']:
        x_test, y_test = augment_data(x=x_test, y=y_test, flipud=global_vars['FLIP_UPSIDE_DOWN'],
                                      fliplr=global_vars['FLIP_LEFT_TO_RIGHT'])
    return x_train, x_val, x_test, y_train, y_val, y_test


def unlabeled_data_reader(global_vars: Dict[str, Any]) -> np.ndarray:
    """ This function reads data and returns ready to use data for testing. """
    x = list()
    images_dir = Path(global_vars['TEST_PATH'])
    for image in sorted(images_dir.glob('*.jpg')):
        img = imread(image)
        if img.shape[0] != 230:
            continue
        x.append(img)
    x = np.array(x)

    # Normalize data
    if global_vars['NORMALIZE_PIXEL_VALUES']:
        x = normalize_data(x)
    return x
