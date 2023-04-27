from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784',version=1,cache=True)
x,y=mnist["data"],mnist["target"]
print(x.shape)
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
some_digit=x.to_numpy()[26000]
some_digit_image=some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()
x_train,x_test,y_train,y_test=x[:60000],x[60000:],y[:60000],y[60000:]
print(y_train)
import numpy as np
y_train=y_train.astype(np.int8)
print(y_train)
y_train_4=(y_train==4)
print(y_train_4)
y_test_4=(y_test==4)
print(y_test_4)
from sklearn.linear_model import SGDClassifier
sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(x_train,y_train_4)
sgd_clf.predict([some_digit])
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, x_train, y_train_4, cv=3, scoring="accuracy")
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_4, cv=3)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_4, y_train_pred) 
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_4, y_train_pred)
recall_score(y_train_4, y_train_pred) 
from sklearn.metrics import f1_score
f1_score(y_train_4, y_train_pred)