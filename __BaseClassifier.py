import logging
from typing import Any, Dict

import numpy as np

from evals.confusion_matrix import calculate_and_plot_confusion_matrix
from evals.test_predictions import save_images


class __BaseClassifierClass:

    def __init__(self):
        return

    def define(self, image_height: int, image_width: int, channels: int, num_classes: int):
        raise NotImplementedError("Please implement the define function!")

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
              model_save_path: str):
        raise NotImplementedError("Please implement the train function!")

    def evaluate(self, x: np.ndarray, y: np.ndarray, model_load_path: str):
        raise NotImplementedError("Please implement the evaluate function!")

    def save_model(self, global_vars: Dict[str, Any], accuracy_list: list):
        raise NotImplementedError("Please implement the save_model function!")

    def predict(self, x: np.ndarray, model_load_path: str):
        raise NotImplementedError("Please implement the test function!")

    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray, classes: np.ndarray, title: str, plot_fig: bool = True):
        logging.info(f'Plotting {title} confusion matrix...\n')
        calculate_and_plot_confusion_matrix(confusion_matrix=confusion_matrix, classes=classes, plot_fig=plot_fig,
                                            title=title)
        calculate_and_plot_confusion_matrix(confusion_matrix=confusion_matrix, classes=classes, plot_fig=plot_fig,
                                            normalize=True, title=f'{title}_norm_')
        return None

    @staticmethod
    def save_test_predictions(x: np.ndarray, predictions: np.ndarray, save_path: str):
        save_images(x=x, predictions=predictions, save_path=save_path)
        return None
