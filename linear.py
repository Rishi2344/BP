import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('/content/salary_data.csv')
import matplotlib.pyplot as plt
df.plot(x="YearsExperience",y="Salary",style='o')
plt.xlabel('Experience')
plt.ylabel('Salaries')
plt.show()
X=pd.DataFrame(df['YearsExperience'])
y=pd.DataFrame(df['Salary'])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=1/3,
random_state=1)
lr = LinearRegression().fit(X_train, y_train) 
y_pred = lr.predict(X_test) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
y_pred
y_test
from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


----output:
image:
(20, 1)
(10, 1)
(20, 1)
(10, 1)
[26137.2400142]
[[9158.13919873]]
Mean Absolute Error: 5049.818093659747
Mean squared Error: 37496296.61879842
Root Mean Squared Error 6123.421969683162