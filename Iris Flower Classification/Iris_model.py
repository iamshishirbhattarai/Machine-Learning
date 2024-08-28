import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Loading and preparing dataset
iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_target = pd.DataFrame(iris.target, columns=['Class'])
iris_df = pd.concat([iris_data, iris_target], axis=1)

# splitting the dataset into training and testing datas
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.2, random_state=42)


def model():
    rf_clf = RandomForestClassifier(random_state=42)
    return rf_clf.fit(X_train, y_train)




