import itertools

import numpy as np
from matplotlib import pyplot as plt


def calculate_and_plot_confusion_matrix(confusion_matrix: np.ndarray, classes: np.ndarray, plot_fig: bool,
                                        normalize: bool = False, title: str = 'Confusion matrix'):
    """
    This function plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks_x = np.arange(len(classes))
    tick_marks_y = np.arange(len(classes))
    plt.xticks(tick_marks_x, classes, rotation=0)
    plt.yticks(tick_marks_y, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show(block=plot_fig)
    plt.close()
