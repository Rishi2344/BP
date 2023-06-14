import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/content/salary_data.csv")

target = df.iloc[:, 1].values

train = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.20)

X_train= X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

m1 = LinearRegression()
m2 = xgb.XGBRegressor()
m3 = RandomForestRegressor()

m1.fit(X_train, y_train)
m2.fit(X_train, y_train)
m3.fit(X_train, y_train)

p1 = m1.predict(X_test)
p2 = m2.predict(X_test)
p3 = m3.predict(X_test)

pf = (p1+p2+p3)/3.0

print("Average Result:",pf)



print("MSE in Average method:",mean_squared_error(y_test, pf))

