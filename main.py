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
