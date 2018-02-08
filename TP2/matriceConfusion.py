from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree
from sklearn.model_selection import train_test_split
import random 

X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X,
Y,random_state=0)

clf=tree.DecisionTreeClassifier()
clf=clf.fit(X_train, Y_train)
Y_pred =clf.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

'''
print (Y_test)
print (Y_pred)
print(iris.feature_names)
'''
