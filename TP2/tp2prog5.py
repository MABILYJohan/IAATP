from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree
from sklearn.model_selection import train_test_split
import random 
from sklearn.metrics import confusion_matrix
import math


def ex_5():
	X,Y=make_classification(n_samples=100000, n_informative=15, \
	n_features=20, n_classes=3)
	
	X_1,X_2,Y_1,Y_2=\
	train_test_split(X,Y,test_size=0.95,random_state=random.seed())
	
	X_app, X_test, Y_app, Y_test = \
	train_test_split(X_1, Y_1, test_size=0.2, random_state=random.seed())
	
	clf=tree.DecisionTreeClassifier()
	clf=clf.fit(X_app, Y_app)
	print ('app')
	N=1000
	nbErr=0;
	i = 1
	while i < 100:
		clf=tree.DecisionTreeClassifier(max_leaf_nodes=500*i) #I
		i +=1
		clf = clf.fit(X_app,Y_app)
		e = 1-clf.score(X_test,Y_test)
		print("%6.4f" %e)
		binf = e - 1.96 * math.sqrt((e * (1-e))/N)
		bsup = e + 1.96 * math.sqrt((e * (1-e))/N)
		"""
		print ("intervalle de confiance I")
		print ("	binf = %6.4f" %binf)
		print ("	bsup = %6.4f" %bsup)
		print ("Erreur estimee f sur X_2, Y_2")
		"""
		f = 1-clf.score(X_2,Y_2)
		#print ("	f = %6.4f" %f)
		if f<binf or f>bsup:
			nbErr+=1
	print ("nb de fois ou f n'appartient pas a I: %d" %nbErr)

ex_5()
