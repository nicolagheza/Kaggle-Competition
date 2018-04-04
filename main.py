import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Load data
train_df = pd.read_csv('./competitions/ken3450/train.csv')
test_df  = pd.read_csv('./competitions/ken3450/test.csv')

X_data = train_df.drop("shares", axis=1)
y_data = train_df["shares"]

X_train, y_train, X_test, y_test = train_test_split(X_data, y_data,
													test_size=0.30, 
													random_state=42)
#Features scaling
scaler = StandardScaler()
test = scaler.fit_transform(X_train)
print (test)