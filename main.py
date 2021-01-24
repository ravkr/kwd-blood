import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

# Wczytywanie danych
input_data = pd.read_csv('data/transfusion.data')
# input_data = pd.read_csv('data/transfusion2.data')

# Analiza danych
sns.pairplot(input_data)
print(input_data.head())
print(input_data.shape)
sns.heatmap(input_data.corr())
plt.show()
print(tabulate(input_data.corr(), headers='keys', tablefmt='psql'))
print(tabulate(input_data.describe(), headers='keys', tablefmt='psql'))

# F (Frequency - total number of donation)
# M (Monetary - total blood donated in c.c.) - mililitry krwi
# ponieważ Monetary wynika z Frequency, to pozbywamy się tego drugiego

input_data = input_data.drop(columns="Monetary (c.c. blood)")

input_features = input_data.iloc[:, :-1]
target = input_data.iloc[:, -1:]
# print(input_features)
# print(target)
# plt.show()
# print(input_data[54])


# Normalizacja
# if necessary to scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(input_features)


train_data, test_data, train_target, test_target = \
    train_test_split(scaled_data, target.T.squeeze().to_numpy(), test_size=0.1, random_state=10)

# Uczenie
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression(max_iter=100)
logistic_regression.fit(train_data, train_target)

index = 9
prediction = logistic_regression.predict(test_data[index, :].reshape(1, -1))
print(f"Model predicted for \"{index}\" value {prediction}")
print(f"Real value for \"{index}\" is {test_target[index]}")


# Sprawdzanie poprawności
acc = accuracy_score(test_target, logistic_regression.predict(test_data))
print("Model accuracy is {0:0.2f}".format(acc))

conf_matrix = confusion_matrix(test_target, logistic_regression.predict(test_data))
print(conf_matrix)


cv_results = cross_validate(logistic_regression, train_data, train_target, scoring=('accuracy'), cv=10)
accuracy = cv_results['test_score'].mean()
print("Model accuracy via cross-validation is {0:0.2f}".format(accuracy))


from sklearn.metrics import accuracy_score, confusion_matrix

for depth in range(1, 10):
    decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=0)
    decision_tree.fit(train_data, train_target)

    print(f"depth = {depth}")

    print(accuracy_score(test_target, decision_tree.predict(test_data)))
    print(confusion_matrix(test_target, decision_tree.predict(test_data)))

    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier(max_depth=depth, criterion='entropy', random_state=0)
    random_forest.fit(train_data, train_target)

    print(accuracy_score(test_target, random_forest.predict(test_data)))
    print(confusion_matrix(test_target, random_forest.predict(test_data)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from keras.layers import Dropout

#set seed for reproduction purpose
from numpy.random import seed
seed(1)

import random as rn
rn.seed(12345)

import tensorflow as tf
tf.random.set_seed(1234)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
neural_model = Sequential([
    Dense(8, input_shape=(3,), activation="relu"),
    Dense(5, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
])


#show summary of a model
neural_model.summary()

neural_model.compile(SGD(lr = .003), "binary_crossentropy", metrics=["accuracy"])

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

plt.plot(run_hist.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.title("Train loss and validation error")
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('Error')
plt.grid()
plt.show()
