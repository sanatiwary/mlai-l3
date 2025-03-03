import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

columns = ["age", "workClass", "fnlwgt", "education", "educationNum", "maritalStatus", "occupation", "relationship", "race", "sex", "capitalGain", "capitalLoss", "hoursPerWeek", "nativeCountry", "income"]
data = pd.read_csv("adult.csv", names=columns)

numericData = data[["age", "educationNum", "hoursPerWeek", "capitalGain"]]

x = numericData[["age", "educationNum", "hoursPerWeek"]]
y = numericData["capitalGain"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

linModel = LinearRegression()
linModel.fit(xTrain, yTrain)

yTestPredict = linModel.predict(xTest)
rmseLinModel = np.sqrt(mean_squared_error(yTest, yTestPredict))
print("rmse in case of linear regression: ", rmseLinModel)

polyFeature = PolynomialFeatures(degree=2)
xTrainPoly = polyFeature.fit_transform(xTrain)

polyModel = LinearRegression()
polyModel.fit(xTrainPoly, yTrain)

xTestPoly = polyFeature.fit_transform(xTest)
yTestPredictPoly = polyModel.predict(xTestPoly)

rmsePolyModel = np.sqrt(mean_squared_error(yTest, yTestPredictPoly))
print("the rmse in case of polynomial regression: ", rmsePolyModel)
