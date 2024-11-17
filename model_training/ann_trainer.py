import csv

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.layers import Dense
from keras.api.models import Sequential
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    xtrain = pd.read_csv("../data/X_train.csv")
    xtest = pd.read_csv("../data/X_test.csv")
    ytrain = pd.read_csv("../data/y_train.csv")
    ytest = pd.read_csv("../data/y_test.csv")

    # getting scaled features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(xtrain)
    X_test_scaled = scaler.transform(xtest)

    # took about 25 minutes to run on Colab. Locally, could be take more time.
    # ann_models.csv file has this data.

    eps = [75, 100, 125, 150, 175, 200, 225, 250]
    neurons = [32, 64, 128, 256, 512]
    metdict = {}

    def metrics(eps, neu, ytest, ypred):
        return accuracy_score(ytest, ypred), f1_score(
            ytest, ypred), recall_score(ytest, ypred), precision_score(
                ytest, ypred), matthews_corrcoef(ytest,
                                                 ypred), cohen_kappa_score(
                                                     ytest, ypred)

    for i in range(len(eps)):
        for j in range(len(neurons)):
            print(f'for {eps[i]}, {neurons[j]}, the results are: ')
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
            y_pred = model.predict(X_test_scaled)
            ypred = []
            for k in y_pred:
                if k > 0.5:
                    ypred.append(1)
                else:
                    ypred.append(0)

            ypred = np.asarray(ypred)
            acc, f1, rec, pre, mcc, coh = metrics(eps[i], neurons[j], ytest,
                                                  ypred)
            val = [acc, f1, rec, pre, mcc, coh]
            metdict[f'for {eps[i]}, {neurons[j]}, the results are: '] = val
            print(val)

    w = csv.writer(open("ann_models.csv", "w"))
    for key, val in metdict.items():
        w.writerow([key, val])

    # ann_results.csv has the metrics of these models.

    tf.keras.backend.clear_session()

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

    model.compile(optimizer="sgd",
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.Recall(), 'accuracy'])
    model1.compile(optimizer="sgd",
                   loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.Recall(), 'accuracy'])
    model2.compile(optimizer="sgd",
                   loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.Recall(), 'accuracy'])
    model3.compile(optimizer="sgd",
                   loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.Recall(), 'accuracy'])
    model4.compile(optimizer="sgd",
                   loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.Recall(), 'accuracy'])
    model5.compile(optimizer="sgd",
                   loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.Recall(), 'accuracy'])
    model6.compile(optimizer="sgd",
                   loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.Recall(), 'accuracy'])
    model7.compile(optimizer="sgd",
                   loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.Recall(), 'accuracy'])
    model8.compile(optimizer="sgd",
                   loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.Recall(), 'accuracy'])

    history = model.fit(X_train_scaled,
                        ytrain,
                        validation_split=0.25,
                        epochs=250,
                        verbose=1)
    history1 = model1.fit(X_train_scaled,
                          ytrain,
                          validation_split=0.25,
                          epochs=250,
                          verbose=1)
    history2 = model2.fit(X_train_scaled,
                          ytrain,
                          validation_split=0.25,
                          epochs=250,
                          verbose=1)
    history3 = model3.fit(X_train_scaled,
                          ytrain,
                          validation_split=0.25,
                          epochs=250,
                          verbose=1)
    history4 = model4.fit(X_train_scaled,
                          ytrain,
                          validation_split=0.25,
                          epochs=250,
                          verbose=1)
    history5 = model5.fit(X_train_scaled,
                          ytrain,
                          validation_split=0.25,
                          epochs=250,
                          verbose=1)
    history6 = model6.fit(X_train_scaled,
                          ytrain,
                          validation_split=0.25,
                          epochs=250,
                          verbose=1)
    history7 = model7.fit(X_train_scaled,
                          ytrain,
                          validation_split=0.25,
                          epochs=250,
                          verbose=1)
    history8 = model8.fit(X_train_scaled,
                          ytrain,
                          validation_split=0.25,
                          epochs=250,
                          verbose=1)

    w = csv.writer(open("ann_results.csv", "w"))
    for key, val in metdict.items():
        w.writerow([key, val])
