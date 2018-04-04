import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

#Load data
train_df = pd.read_csv('./competitions/ken3450/train.csv')
test_df  = pd.read_csv('./competitions/ken3450/test.csv')

X_train = train_df.drop("shares", axis=1)
y_train = train_df["shares"]

pipeline = make_pipeline(StandardScaler(), MLPRegressor(activation='relu', 
														verbose=True, max_iter=500))
mlp_model=pipeline.fit(X_train, y_train)

pred = mlp_model.predict(test_df)

df = pd.DataFrame({
        'id': pd.Series(range(1,9912)),
        'shares': pred
})

df.to_csv("sol.csv", index=False)
