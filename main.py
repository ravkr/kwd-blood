import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.layers import Dropout
from numpy.random import seed
import random as rn
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

PATH_ORIGINAL = 'data/transfusion.data'
PATH_MODIFIED = 'data/transfusion2.data'

paths = [PATH_ORIGINAL, PATH_MODIFIED]


def load_and_analyze_data(path, verbose=False):
    # Wczytywanie danych
    input_data = pd.read_csv(path)
    if verbose:
        # Analiza danych
        sns.pairplot(input_data)
        plt.show()
        print(input_data.head())
        print(input_data.shape)
        sns.heatmap(input_data.corr())
        plt.show()
        columns_copy = input_data.columns
        input_data.columns = ['R', 'F', 'M', 'T', 'D']
        print(tabulate(input_data.corr(), headers='keys', tablefmt='psql'))
        print(tabulate(input_data.describe(), headers='keys', tablefmt='psql'))
        input_data.columns = columns_copy
    return input_data


def preprocess_data(input_data):
    # F (Frequency - total number of donation)
    # M (Monetary - total blood donated in c.c.) - mililitry krwi
    # ponieważ Monetary wynika z Frequency, to pozbywamy się tego drugiego

    input_data = input_data.drop(columns="Monetary (c.c. blood)")

    input_features = input_data.iloc[:, :-1]
    target = input_data.iloc[:, -1:]

    # Normalizacja
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_features)

    train_data, test_data, train_target, test_target = \
        train_test_split(scaled_data, target.T.squeeze().to_numpy(), test_size=0.1, random_state=10)
    return train_data, test_data, train_target, test_target


def train_and_score_model(train_data, test_data, train_target, test_target, model, model_params, x):
    model_instance = model(**model_params)
    model_instance.fit(train_data, train_target)

    # Sprawdzanie poprawności
    acc = accuracy_score(test_target, model_instance.predict(test_data))
    # print("Model accuracy is {0:0.2f}".format(acc))

    conf_matrix = confusion_matrix(test_target, model_instance.predict(test_data))
    # print(conf_matrix)

    cv_results = cross_validate(model_instance, train_data, train_target, scoring=('accuracy'), cv=10)
    accuracy = cv_results['test_score'].mean()
    # print("Model accuracy via cross-validation is {0:0.2f}".format(accuracy))

    return [x, [acc, accuracy]], [x, conf_matrix]


for path in paths:
    input_data = load_and_analyze_data(path)
    train_data, test_data, train_target, test_target = preprocess_data(input_data)

    models = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]

    models_params = [{'max_iter': range(100, 1001, 100), 'random_state': 42},
                     {'criterion': 'entropy', 'max_depth': range(1, 11), 'random_state': 42},
                     {'criterion': 'entropy', 'max_depth': range(1, 11), 'random_state': 42}]
    models_accuracy = {}
    models_conf_matrices = {}
    for i, model in enumerate(models):
        model_name = model().__class__.__name__
        models_accuracy[model_name] = []
        models_conf_matrices[model_name] = []
        params = models_params[i]
        iterations = range(1)
        if 'max_iter' in params:
            iterations = params['max_iter']
        elif 'max_depth' in params:
            iterations = params['max_depth']
        for j in iterations:
            if 'max_iter' in params:
                params['max_iter'] = j
            elif 'max_depth' in params:
                params['max_depth'] = j
            current_accuracy, current_conf_matrix = train_and_score_model(train_data, test_data, train_target,
                                                                          test_target, model, params, j)
            models_accuracy[model_name].append(current_accuracy)
            models_conf_matrices[model_name].append(current_conf_matrix)

    print(models_accuracy)
    print(models_conf_matrices)

    for key, value in models_accuracy.items():
        indexes = [element[0] for element in value]
        accuracy = [element[1][0] for element in value]
        cv_accuracy = [element[1][1] for element in value]

        plt.plot(indexes, accuracy, 'r', marker='.', label="Accuracy")
        plt.plot(indexes, cv_accuracy, 'b', marker='.', label="CV Accuracy")
        plt.title(key + " model accuracy:")
        plt.legend()
        if 'Regression' in key:
            plt.xlabel('Iterations'), plt.ylabel('Accuracy')
        elif 'Tree' in key or 'Forest' in key:
            plt.xlabel('Depth'), plt.ylabel('Accuracy')
        plt.grid()
        plt.show()

    seed(1)
    rn.seed(12345)
    tf.random.set_seed(1234)

    neural_model = Sequential([
        Dense(8, input_shape=(3,), activation="relu"),
        Dense(5, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid")
    ])

    # show summary of a model
    neural_model.summary()

    neural_model.compile(SGD(lr=.003), "binary_crossentropy", metrics=["accuracy"])

    np.random.seed(0)
    run_hist = neural_model.fit(train_data, train_target, epochs=1000,
                                validation_data=(test_data, test_target),
                                verbose=True, shuffle=True)

    print("Training neural network...\n")

    print('Accuracy over training data is ',
          accuracy_score(train_target, neural_model.predict_classes(train_data)))

    print('Accuracy over testing data is ',
          accuracy_score(test_target, neural_model.predict_classes(test_data)))

    conf_matrix = confusion_matrix(test_target, neural_model.predict_classes(test_data))
    print(conf_matrix)

    plt.plot(run_hist.history["accuracy"], 'r', marker='.', label="Train Accuracy")
    plt.plot(run_hist.history["val_accuracy"], 'b', marker='.', label="Validation Accuracy")
    plt.title("Train and validation accuracy")
    plt.legend()
    plt.xlabel('Epoch'), plt.ylabel('Error')
    plt.grid()
    plt.show()

# STARY KOD

# Uczenie


# logistic_regression = LogisticRegression(max_iter=100)
# logistic_regression.fit(train_data, train_target)
#
# index = 9
# prediction = logistic_regression.predict(test_data[index, :].reshape(1, -1))
# print(f"Model predicted for \"{index}\" value {prediction}")
# print(f"Real value for \"{index}\" is {test_target[index]}")
#
# # Sprawdzanie poprawności
# acc = accuracy_score(test_target, logistic_regression.predict(test_data))
# print("Model accuracy is {0:0.2f}".format(acc))
#
# conf_matrix = confusion_matrix(test_target, logistic_regression.predict(test_data))
# print(conf_matrix)
#
# cv_results = cross_validate(logistic_regression, train_data, train_target, scoring=('accuracy'), cv=10)
# accuracy = cv_results['test_score'].mean()
# print("Model accuracy via cross-validation is {0:0.2f}".format(accuracy))
#
#
#
# for depth in range(1, 10):
#     decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=0)
#     decision_tree.fit(train_data, train_target)
#
#     print(f"depth = {depth}")
#
#     print(accuracy_score(test_target, decision_tree.predict(test_data)))
#     print(confusion_matrix(test_target, decision_tree.predict(test_data)))
#
#
#
#     random_forest = RandomForestClassifier(max_depth=depth, criterion='entropy', random_state=0)
#     random_forest.fit(train_data, train_target)
#
#     print(accuracy_score(test_target, random_forest.predict(test_data)))
#     print(confusion_matrix(test_target, random_forest.predict(test_data)))
