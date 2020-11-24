import logging
from pathlib import Path
from unittest import TestCase

import numpy as np

from class_CNN import CNN
from data_utils import data_reader, unlabeled_data_reader
from setup import GLOBAL_VARS


class TestCNN1D(TestCase):

    @staticmethod
    def setup_train_assertions():
        # Path Assertions
        assert (Path(GLOBAL_VARS['DATASET_ROOT']).exists()), f"{GLOBAL_VARS['DATASET_ROOT']} doesn't exist"
        assert (Path(GLOBAL_VARS['CLASS1_PATH']).exists()), f"{GLOBAL_VARS['CLASS1_PATH']} doesn't exist"
        assert (Path(GLOBAL_VARS['CLASS2_PATH']).exists()), f"{GLOBAL_VARS['CLASS2_PATH']} doesn't exist"

    @staticmethod
    def setup_test_assertions():
        # Path Assertions
        assert (Path(GLOBAL_VARS['TEST_PATH']).exists()), f"{GLOBAL_VARS['TEST_PATH']} doesn't exist"
        assert (Path(GLOBAL_VARS['TRAINED_MODEL']).exists()), f"{GLOBAL_VARS['TRAINED_MODEL']} doesn't exist"

    def setup_CNN(self):
        """ Reading data for training and testing """
        self.setup_train_assertions()
        x_train, x_val, x_test, y_train, y_val, y_test = data_reader(GLOBAL_VARS)
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = len(np.unique(y_test))

    def setup_predict_CNN(self):
        """ Reading data for testing """
        self.setup_test_assertions()
        x_test = unlabeled_data_reader(global_vars=GLOBAL_VARS)
        self.x_test = x_test

    def test_train_CNN(self):
        """ Training, Testing and Evaluating """
        clf = CNN()
        self.setup_CNN()

        clf.define(image_height=230, image_width=130, channels=3, num_classes=self.num_classes)
        clf.train(x_train=self.x_train, y_train=self.y_train, x_val=self.x_val, y_val=self.y_val,
                  model_save_path=GLOBAL_VARS["TRAINING_MODELS"])
        accuracy, confusion_matrix = clf.evaluate(x=self.x_test, y=self.y_test,
                                                  model_load_path=GLOBAL_VARS["TRAINING_MODELS"])
        clf.save_model(accuracy_list=accuracy, global_vars=GLOBAL_VARS)
        clf.plot_confusion_matrix(confusion_matrix=confusion_matrix[0], classes=np.unique(self.y_test),
                                  title=f'{type(clf).__name__} __ Validation Best Model', plot_fig=True)
        clf.plot_confusion_matrix(confusion_matrix=confusion_matrix[1], classes=np.unique(self.y_test),
                                  title=f'{type(clf).__name__} __ Last Epoch Model', plot_fig=True)

    def test_predict_CNN(self):
        """ Testing only """
        clf = CNN()
        self.setup_predict_CNN()

        predictions = clf.predict(x=self.x_test, model_load_path=GLOBAL_VARS["TRAINED_MODEL"])
        logging.info(predictions)
        clf.save_test_predictions(x=self.x_test, predictions=predictions, save_path=GLOBAL_VARS["PREDICTED_LABELS"])
