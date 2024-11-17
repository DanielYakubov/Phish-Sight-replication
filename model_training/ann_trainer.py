import csv
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.layers import Dense
from keras.api.models import Sequential
from metrics_utils import get_metrics
from sklearn.preprocessing import StandardScaler


def _write_model_results_to_file(file_name: str, metdict: Dict[str,
                                                               float]) -> None:
    """Writes the results of a model to the specified file name

    Args:
        metdict (Dict[str, float]): a dictionary containing the values for the metrics

    Returns:
        None, writes a file
    """
    w = csv.writer(open(file_name, "w"))
    for key, val in metdict.items():
        w.writerow([key, val])


def ann_hyperparameter_search(eps: List[int], neurons: List[int]) -> None:
    """Performs a grid search for models over epochs and neurons

    Args:
        eps (List[int]): A list of epoch number options to search through
        neurons (List[int]): A list of neuron options to search through

    Returns:
        None, writes a file
    """
    metdict = {}
    for i in range(len(eps)):
        for j in range(len(neurons)):
            logging.info(f'for {eps[i]}, {neurons[j]}, the results are: ')
            model = Sequential()
            model.add(Dense(neurons[j], activation="relu"))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='sgd',
                          loss="binary_crossentropy",
                          metrics=['accuracy'])
            model.fit(X_train_scaled,
                      ytrain,
                      validation_split=0.25,
                      epochs=eps[i],
                      verbose=0)
            y_hat = model.predict(X_test_scaled)
            ypred = []
            for k in y_hat:
                if k > 0.5:
                    ypred.append(1)
                else:
                    ypred.append(0)

            ypred = np.asarray(ypred)
            metdict[
                f'for {eps[i]}, {neurons[j]}, the results are: '] = get_metrics(
                    ytest, ypred)
    _write_model_results_to_file("..data/experiment_results/ann_models.csv",
                                 metdict)


if __name__ == '__main__':
    xtrain = pd.read_csv("../data/X_train.csv")
    xtest = pd.read_csv("../data/X_test.csv")
    ytrain = pd.read_csv("../data/y_train.csv")
    ytest = pd.read_csv("../data/y_test.csv")

    # getting scaled features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(xtrain)
    X_test_scaled = scaler.transform(xtest)

    # eps = [75, 100, 125, 150, 175, 200, 225, 250]
    # neurons = [32, 64, 128, 256, 512]
    # ann_hyperparameter_search(eps, neurons)

    tf.keras.backend.clear_session()

    # initialization of models that will be used
    model = Sequential()
    model.add(Dense(1, activation='sigmoid'))

    model1 = Sequential()
    model1.add(Dense(32, activation='relu'))
    model1.add(Dense(1, activation='sigmoid'))

    model2 = Sequential()
    model2.add(Dense(64, activation='relu'))
    model2.add(Dense(1, activation='sigmoid'))

    model3 = Sequential()
    model3.add(Dense(128, activation='relu'))
    model3.add(Dense(1, activation='sigmoid'))

    model4 = Sequential()
    model4.add(Dense(256, activation='relu'))
    model4.add(Dense(1, activation='sigmoid'))

    model5 = Sequential()
    model5.add(Dense(512, activation='relu'))
    model5.add(Dense(1, activation='sigmoid'))

    model6 = Sequential()
    model6.add(Dense(1024, activation='relu'))
    model6.add(Dense(1, activation='sigmoid'))

    model7 = Sequential()
    model7.add(Dense(256, activation='relu'))
    model7.add(Dense(256, activation='relu'))
    model7.add(Dense(1, activation='sigmoid'))

    model8 = Sequential()
    model8.add(Dense(512, activation='relu'))
    model8.add(Dense(512, activation='relu'))
    model8.add(Dense(1, activation='sigmoid'))

    histories = []
    metdict = {}
    for i, model_ in enumerate(
        [model, model1, model2, model3, model4, model5, model6, model7,
         model8]):
        model_.compile(optimizer="sgd",
                       loss="binary_crossentropy",
                       metrics=[tf.keras.metrics.Recall(), 'accuracy'])
        histories.append(
            model_.fit(X_train_scaled,
                       ytrain,
                       validation_split=0.25,
                       epochs=250,
                       verbose=1))
        model.fit(X_train_scaled,
                  ytrain,
                  validation_split=0.25,
                  epochs=250,
                  verbose=0)
        y_hat = model.predict(X_test_scaled)
        ypred = []
        for k in y_hat:
            if k > 0.5:
                ypred.append(1)
            else:
                ypred.append(0)
        ypred = np.asarray(ypred)
        metdict[f'model_{i}'] = get_metrics(ytest, ypred)
    _write_model_results_to_file("../data/experiment_results/ann_results.csv",
                                 metdict)
