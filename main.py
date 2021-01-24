import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

# Wczytywanie danych
input_data = pd.read_csv('data/transfusion.data')

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





