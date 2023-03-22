from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import random
data_iris = load_iris()
label_target = data_iris.target_names
print()
print("Sample Data from Iris Dataset")
print("*"*30)
for i in range(10):
  rn = random.randint(0,120)
  print(data_iris.data[rn],"===>",label_target[data_iris.target[rn]])
X = data_iris.data
y = data_iris.target

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size = 0.3, random_state=1)
print("The Training dataset length: ",len(X_train))
print("The Testing dataset length: ",len(X_test))
try:
  nn = int(input("Enter number of neighbors :"))
  knn = KNeighborsClassifier(nn)
  knn.fit(X_train, y_train)
  print("The Score is :",knn.score(X_test, y_test))
  test_data = input("Enter Test Data :").split(",")
  for i in range(len(test_data)):
    test_data[i] = float(test_data[i])
  print()
  v = knn.predict([test_data])
  print("Predicted output is :",label_target[v])
except:
  print("Please supply valid input......")

----Output:
Sample Data from Iris Dataset
******************************
[5.1 3.3 1.7 0.5] ===> setosa
[6.1 3.  4.6 1.4] ===> versicolor
[5.7 2.9 4.2 1.3] ===> versicolor
[6.4 3.2 4.5 1.5] ===> versicolor
[4.5 2.3 1.3 0.3] ===> setosa
[5.4 3.9 1.3 0.4] ===> setosa
[6.1 2.8 4.  1.3] ===> versicolor
[5.  3.5 1.6 0.6] ===> setosa
[4.9 2.5 4.5 1.7] ===> virginica
[4.9 3.1 1.5 0.1] ===> setosa
The Training dataset length:  105
The Testing dataset length:  45
Enter number of neighbors :3
The Score is : 0.9777777777777777
Enter Test Data :1,3,5,7

Predicted output is : ['virginica']