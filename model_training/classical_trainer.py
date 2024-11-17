"""This module contains the code to train classical machine learning models on the dataset"""

import logging
from typing import Any, Dict, Protocol

import joblib
import numpy as np
import pandas as pd
from metrics_utils import get_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ScikitModel(Protocol):
    # https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
    def fit(self, X, y, sample_weight=None):
        ...

    def predict(self, X):
        ...

    def score(self, X, y, sample_weight=None):
        ...

    def set_params(self, **params):
        ...


def hyperparameter_search(model: ScikitModel, params: Dict[str, Any],
                          train_x: np.array,
                          y_train: np.array) -> Dict[str, Any]:
    """Performs hyperparameter search using gridsearch across a params dict.

    Args:
        model (sklearn.BaseEstimator): The model to be trained
        params (Dict[str, Any]): The dictionary containing the parameters for training.
        scaling (str): specify if scaling is needed for the data. This depends on the model.
        train_x (np.array): the training input data
        y_train (np.array): the training labels
    Returns:
        bp (Dict[str, Any]): The best parameters for the model
    """
    logging.info(f"Starting hyperparameter search for {model.__name__}")
    gs_clf = GridSearchCV(estimator=model(),
                          param_grid=params,
                          cv=5,
                          verbose=0,
                          scoring='recall')  # we want to maximize recall
    gs_clf.fit(train_x, y_train)
    bp = gs_clf.best_params_
    logging.info(f"Best parameters found {bp} for model {model.__name__}")
    return bp


def train_and_get_pred(model: ScikitModel, best_params: Dict[str, Any],
                       x_train: np.array, y_train: np.array,
                       x_test: np.array) -> np.array:
    """Train the using the best parameters and return predictions

    Args:
        model (sklearn.BaseEstimator): The model type we are training
        best_params (Dict[str, Any]): The collection of the best parameters for the model
        x_train (np.array): The input training data
        y_train (np.array): The label training data
        x_test (np.array): The input to predict on

    Returns:
         y_hat (np.array): The predicted labels
    """
    best_model = model(**best_params)
    best_model.fit(x_train, y_train)
    joblib.dump(best_model, f"../data/saved_models/{model.__name__}.pkl")
    y_hat = best_model.predict(x_test)
    return y_hat


if __name__ == "__main__":
    # reading in data
    X_train = pd.read_csv("../data/X_train.csv").to_numpy()
    X_test = pd.read_csv("../data/X_test.csv").to_numpy()
    y_train = pd.read_csv("../data/y_train.csv").to_numpy().ravel()
    y_test = pd.read_csv("../data/y_test.csv").to_numpy().ravel()

    # getting scaled features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # setting up params for models
    logistic_params = {
        "solver": ["liblinear"],  # small dataset - liblinear makes sense
        "C": [1.0, 0.1, 0.01, 0.001, 0.0001],
        "penalty": ["l1", "l2"]
    }
    KNN_params = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "weights": ['uniform', 'distance'],
        "p": [1, 2]
    }
    SVC_params = {
        "C": [1.0, 0.1, 0.01, 0.001, 0.0001],
        "gamma": ["scale", "auto"],
        "degree": [1, 2, 3, 4],
        "kernel": ["linear", "poly", "rbf", "sigmoid"]
    }
    DT_params = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 3, 5, 7, 9]
    }
    RF_params = {
        "n_estimators": [25, 50, 100, 150],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 3, 5, 7, 9]
    }

    # models and params and scaling for easy looping
    model_and_params = [
        (LogisticRegression, logistic_params, True),
        (GaussianNB, {}, True),  # NB doesn't really need tuning
        (SVC, SVC_params, True),
        (DecisionTreeClassifier, DT_params, True),
        (RandomForestClassifier, RF_params, True),
        (KNeighborsClassifier, KNN_params, False)
    ]

    metrics = []
    for model, params, scaling in model_and_params:
        if scaling:
            inloop_train_X = X_train_scaled
            inloop_test_X = X_test_scaled
        else:
            inloop_train_X = X_train
            inloop_test_X = X_test
        best_params = hyperparameter_search(model, params, inloop_train_X,
                                            y_train)
        y_hat = train_and_get_pred(model, best_params, inloop_train_X, y_train,
                                   inloop_test_X)
        metrics.append((model.__name__, *get_metrics(y_test, y_hat)))

    metrics = pd.DataFrame(data=metrics,
                           columns=[
                               'Model', "Accuracy", "F1", "Recall", "Precision",
                               "Matthews correlation coefficient (MCC)",
                               "Kappa Coefficient", "FPR"
                           ])
    metrics.to_csv("../data/experiment_results/classical_models.csv")

    # This is the code to load one of these models
    # Decision Tree for example:
    # DT_clf = joblib.load("models/DecisionTreeClassifier.pkl")
