# logistical regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import seaborn as sns

data = pd.read_csv("titanic.csv")
print(data.head())

print(data.isnull().sum())
# age, cabin, embarked

# data preprocessing
print("median of age column %.2f"% (data["Age"].median(skipna=True)))
data["Age"].fillna(data["Age"].median(skipna=True), inplace=True)
print(data.isnull().sum())

print(data["Embarked"].value_counts().idxmax())
data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(), inplace=True)
print(data.isnull().sum())

#dropping unnecessary data
data.drop("Cabin", axis=1, inplace=True)
data.drop("PassengerId", axis=1, inplace=True)
data.drop("Ticket", axis=1, inplace=True)
data.drop("Name", axis=1, inplace=True)

data["TravelAlone"] = np.where((data["SibSp"] + data["Parch"]) > 0, 0, 1)

data.drop("SibSp", axis=1, inplace=True)
data.drop("Parch", axis=1, inplace=True)

print(data.isnull().sum())
print(data.head())

# label encoder replaces string data with numerical values for machine learning processes
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data["Sex"] = le.fit_transform(data["Sex"])
data["Embarked"] = le.fit_transform(data["Embarked"])

print(data.head())

x = data[["Pclass", "Sex", "Age", "Fare", "Embarked", "TravelAlone"]]
y = data["Survived"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(xTrain, yTrain)

yTestPredict = lr.predict(xTest)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(yTest, yTestPredict)

sns.heatmap(matrix, annot=True, fmt="d")
plt.title("matrix")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()