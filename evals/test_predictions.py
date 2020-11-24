from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def save_images(x: np.ndarray, predictions: np.ndarray, save_path: str):
    """
    This function saves the predictions for unlabeled test set.
    """
    for counter, image, label in zip(range(len(predictions)), x, predictions):
        plt.imshow(image)
        plt.title(f'Predicted Label: {label}')
        plt.savefig(Path(save_path, f'{counter}.png'))
        # plt.show()
