import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
import numpy as np

#Load data
train_df = pd.read_csv('./competitions/ken3450/train.csv')
test_df  = pd.read_csv('./competitions/ken3450/test.csv')

X = train_df.drop("shares", axis=1)
y = train_df["shares"]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, test_size=0.3)

print(X_train.shape, type(X_train))
print(y_train.shape, type(y_train))
tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
# df = pd.DataFrame({
#         'id': pd.Series(range(1,9912)),
#         'shares': pred
# })

# df.to_csv("sol.csv", index=False)
