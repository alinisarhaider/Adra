import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

import keras
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, MaxPooling2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from __BaseClassifier import __BaseClassifierClass


class CNN(__BaseClassifierClass):
    """ This class is used to encapsulate all attributes and methods pertaining to CNN Classifier. """
    def __init__(self):
        self.model = None
        self.best_validation_model = None
        super().__init__()

    def define(self, image_height: int, image_width: int, channels: int, num_classes: int):
        """ This function defines and create the architecture of the model """
        logging.info(f'Creating model architecture...')
        model = keras.models.Sequential()
        model.add(Conv2D(64, kernel_size=(4, 4), activation='relu', input_shape=(image_height, image_width, channels)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(8, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        model.summary()
        self.model = model

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
              model_save_path: str):
        """ This function performs model training """
        logging.info(f'Beginning training of created model...')
        # Callback to save the best model during training based upon validation accuracy
        callback = keras.callbacks.ModelCheckpoint(str(model_save_path)+'model-{epoch:03d}-{val_accuracy:.3f}.h5',
                                                   monitor='val_accuracy', verbose=0, save_best_only=True,
                                                   save_weights_only=False, mode='auto', period=1)
        self.model.fit(x_train, to_categorical(y_train), batch_size=32, epochs=99, verbose=1, callbacks=[callback],
                       validation_data=(x_val, to_categorical(y_val)))
        logging.info(f'Model successfully trained...')

    def evaluate(self, x: np.ndarray, y: np.ndarray, model_load_path: str) -> Tuple[List[float], List[np.ndarray]]:
        """ This function perform evaluations on test data after a model is trained. """
        logging.info(f'\nMaking predictions on test data...')
        # Loading the best model saved during training based on validation accuracy.
        best_model_path = sorted(Path(model_load_path).glob('*.h5'), key=os.path.getctime, reverse=True)[0]
        best_model = keras.models.load_model(best_model_path)
        self.best_validation_model = best_model
        accuracy_score_list = list()
        confusion_matrix_list = list()
        for model, type_ in zip([self.best_validation_model, self.model], ['Validation Best Model',
                                                                           'Last Epoch Model']):
            predictions = model.predict(x).argmax(axis=1)
            cnf_matrix = confusion_matrix(y, predictions)
            accuracy = round(accuracy_score(y, predictions) * 100, 3)
            accuracy_score_list.append(accuracy)
            confusion_matrix_list.append(cnf_matrix)
            logging.info(type_)
            logging.info(f'\nConfusion Matrix: \n{cnf_matrix}')
            logging.info(f'Accuracy: {accuracy}')
            logging.info(f'F1: {round(f1_score(y, predictions, average="macro") * 100, 3)}')
            logging.info(f'Precision: {round(precision_score(y, predictions) * 100, 3)}')
            logging.info(f'Recall: {round(recall_score(y, predictions) * 100, 3)}')
        return accuracy_score_list, confusion_matrix_list

    def save_model(self, global_vars: Dict[str, Any], accuracy_list: list):
        """ Save models at the end of training """
        logging.info(f'Saving complete and validation models...')
        self.model.save(f'{global_vars["BEST_BY_VALIDATION_MODELS"]}/{accuracy_list[0]}.h5')
        self.best_validation_model.save(f'{global_vars["LAST_EPOCH_MODELS"]}/{accuracy_list[1]}.h5')

    def predict(self, x: np.ndarray, model_load_path: str) -> np.ndarray:
        """ This function yields predictions on unlabeled test data. """
        logging.info(f'\nMaking predictions on test data...')
        # Loading Pre-trained model.
        best_model_path = sorted(Path(model_load_path).glob('*.h5'), key=os.path.getctime, reverse=True)[0]
        best_model = keras.models.load_model(best_model_path)
        predictions = best_model.predict(x).argmax(axis=1)
        return predictions
