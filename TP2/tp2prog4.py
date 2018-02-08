from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree
from sklearn.model_selection import train_test_split
import random 

def ex4():
	X,Y=make_classification(n_samples=100000,n_features=20,n_informative=15, \
	n_classes=3)
	X_train,X_test,Y_train,Y_test=\
	train_test_split(X,Y,test_size=0.3,random_state=random.seed())
	print ('train')
	i = 1
	while i < 40:
		clf=tree.DecisionTreeClassifier(max_depth = i) #I
		i +=1
		clf = clf.fit(X_train,Y_train)
		print("%6.4f" %clf.score(X_train,Y_train))
	print ('test')
	i = 1
	while i < 40:
		clf=tree.DecisionTreeClassifier(max_depth = i) #I
		i +=1
		clf = clf.fit(X_test,Y_test)
		print("%6.4f" %clf.score(X_test,Y_test))
ex4()
