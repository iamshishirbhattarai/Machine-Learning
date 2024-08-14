#importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

#getting the data
housing_df = pd.read_csv('housing.csv')
print(housing_df.head())

print(housing_df.columns)

