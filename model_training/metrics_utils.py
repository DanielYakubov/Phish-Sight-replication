from typing import Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)


def get_fpr(y_test: np.array, y_hat: np.array) -> float:
    """Calculate the false positive rate for the y labels

    Args:
        y_test (np.array): the gold labels
        y_hat (np.array): the predicted labels

    Returns:
        the false positive rate as a score
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
    fpr = fp / (fp + tn)
    return fpr


def get_metrics(
        y_test: np.array, y_hat: np.array
) -> Tuple[float, float, float, float, float, float, float]:
    """Calculation of the metrics used to evaluate the tradational ML models

    Args:
        y_test (np.array): The gold labels
        y_train (np.array): The predicted labels

    Returns:
        (float) the accuracy score
        (float) the f1 score
        (float) the recall
        (float) the precision
        (float) the matthews correlation coefficient
        (float) the cohen kappa correlation score
        (float) the fprs
    """

    return (accuracy_score(y_test,
                           y_hat), f1_score(y_test,
                                            y_hat), recall_score(y_test, y_hat),
            precision_score(y_test, y_hat), matthews_corrcoef(y_test, y_hat),
            cohen_kappa_score(y_test, y_hat), get_fpr(y_test, y_hat))
