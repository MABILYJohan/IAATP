
import pylab as pl
import random 
import math
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.datasets import load_digits
digits=load_digits()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def get_leaf_with_crossed_validation(X, Y):
	kf=KFold(n_splits=10,shuffle=True)
	scores=[]
	# METTRE A 30
	for k in range(2,30):
		score=0
		clf=tree.DecisionTreeClassifier(max_leaf_nodes=k)
		for learn,test in kf.split(X):
			X_train=X[learn]
			Y_train=Y[learn]
			clf.fit(X_train, Y_train)
			X_test=X[test]
			Y_test=Y[test]
			score = score + clf.score(X_test,Y_test)
		scores.append(score)
	print(["{:4.2f}".format(s) for s in scores])
	k=scores.index(max(scores))+1
	print("meilleure valeur pour k : ", k)
	return k

def train_clf (crit, nbLeaf, X_train, Y_train, X_test, Y_test):
	clf=tree.DecisionTreeClassifier(criterion=crit, max_leaf_nodes=nbLeaf)
	clf = clf.fit(X_train,Y_train)
	
	N=1000
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
	"""
	f = 1-clf.score(X_2,Y_2)
	#print ("	f = %6.4f" %f)
	"""
	return clf

def synthese():
	#print (digits.data[0])
	#print (digits.images[0])
	#print (digits.data[0].reshape(8,8))
	#print (digits.target[0])
	'''
	pl.gray()
	pl.matshow(digits.images[0])
	pl.show()
	'''
	X = digits.data
	Y = digits.target
	
	X_train,X_test,Y_train,Y_test=\
	train_test_split(X,Y,test_size=0.3,random_state=random.seed())
	
	L_train = get_leaf_with_crossed_validation(X_train, Y_train)
	
	# Gini
	clfG = train_clf('gini', L_train, X_train,Y_train,X_test,Y_test)
	clfG = train_clf('entropy', L_train, X_train,Y_train,X_test,Y_test)
	
	
synthese()
