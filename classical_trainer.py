"""This module contains the code to train classical machine learning models on the dataset"""

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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


if __name__ == "__main__":
    # reading in data
    X_train = pd.read_csv("data/X_train.csv").to_numpy()
    X_test = pd.read_csv("data/X_test.csv").to_numpy()
    y_train = pd.read_csv("data/y_train.csv").to_numpy().ravel()
    y_test = pd.read_csv("data/y_test.csv").to_numpy().ravel()

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

    # models and params for easy looping
    model_and_params = [
        (LogisticRegression, logistic_params, 'scaling'),
        (GaussianNB, {}, 'scaling'),  # NB doesn't really need tuning
        (SVC, SVC_params, 'scaling'),
        (DecisionTreeClassifier, DT_params, 'scaling'),
        (RandomForestClassifier, RF_params, 'scaling'),
        (KNeighborsClassifier, KNN_params, 'no_scaling')
    ]

    # setting up lists for metrics
    accs = []
    f1s = []
    recalls = []
    precisions = []
    MCCs = []
    kappas = []
    fprs = []

    # hyperparameter training loop and evaluation
    for model, params, scaling in model_and_params:
        print(f"Starting hyperparameter search for {model.__name__}")
        if scaling == 'scaling':
            inloop_train_X = X_train_scaled
            inloop_test_X = X_test_scaled
        else:
            inloop_train_X = X_train
            inloop_test_X = X_test

        gs_clf = GridSearchCV(estimator=model(),
                              param_grid=params,
                              cv=5,
                              verbose=0,
                              scoring='recall')  # we want to maximize recall
        gs_clf.fit(inloop_train_X, y_train)

        # training the model using the best params
        bp = gs_clf.best_params_
        print(f"Best parameters found {bp}, starting training and eval")
        inloop_model = model(**gs_clf.best_params_)
        inloop_model.fit(inloop_train_X, y_train)
        joblib.dump(inloop_model, f"saved_models/{model.__name__}.pkl")

        # Prediction and eval
        y_hat = inloop_model.predict(inloop_test_X)

        # updating metric lists
        accs.append(accuracy_score(y_test, y_hat))
        f1s.append(f1_score(y_test, y_hat))
        recalls.append(recall_score(y_test, y_hat))
        precisions.append(precision_score(y_test, y_hat))
        MCCs.append(matthews_corrcoef(y_test, y_hat))
        kappas.append(cohen_kappa_score(y_test, y_hat))
        fprs.append(get_fpr(y_test, y_hat))
        print('----')

    # saving metrics
    models = [model.__name__ for model, _, _ in model_and_params
             ]  # getting only the model names
    metrics = pd.DataFrame(data=zip(models, accs, f1s, recalls, precisions,
                                    MCCs, kappas, fprs),
                           columns=[
                               'Model', "Accuracy", "F1", "Recall", "Precision",
                               "Matthews correlation coefficient (MCC)",
                               "Kappa Coefficient", "FPR"
                           ])
    metrics.to_csv("metrics/classical_models.csv")
    print(metrics)

    # This is the code to load one of these models
    # Decision Tree for example:
    # DT_clf = joblib.load("models/DecisionTreeClassifier.pkl")
