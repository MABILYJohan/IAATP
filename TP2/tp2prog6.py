
import pylab as pl
import random 
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.datasets import load_digits
digits=load_digits()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def get_k_with_crossed_validation(X, Y):
	kf=KFold(n_splits=10,shuffle=True)
	scores=[]
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
	
	k = get_k_with_crossed_validation(X, Y)
	
	# Gini
	clf=tree.DecisionTreeClassifier()
	X_train,X_test,Y_train,Y_test=\
	train_test_split(X,Y,test_size=0.3,random_state=random.seed())
	
	# TEMP  --> Choisir meilleur nombre de feuilles par validation croisée.
	# voir (folds) pour sous-échantillons
	#clf=tree.DecisionTreeClassifier(max_leaf_nodes=3)
	
synthese()
