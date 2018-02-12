# Dan LE & Johan MABILY

import pylab as pl
import random 
import math
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn.datasets import load_digits
digits=load_digits()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix





# 2 ARBRES DE DECISION


def tp2prog1():
	clf=tree.DecisionTreeClassifier(max_leaf_nodes=3)
	print (iris.data)
	print (iris.target)
	clf=clf.fit(iris.data, iris.target)
	tree.export_graphviz(clf,out_file="ex1-1.dot")

# exercice 2
'''
Les 2 arbres sont très similaires.
'''

def tp2prog2():
	clf=tree.DecisionTreeClassifier()
	clf=clf.fit(iris.data, iris.target)
	tree.export_graphviz(clf,out_file="tree_gini")
	clf=tree.DecisionTreeClassifier(criterion='entropy')
	clf=clf.fit(iris.data, iris.target)
	tree.export_graphviz(clf,out_file="tree_entropy")

# exercice 3
'''
Au début le classifieur est précis mais toujours avec une marge d'erreur.
A force de relancer avec plus de feuilles, le classifieur est de plus en
plus précis (apprends de mieux en mieux) et se trompe de moins en 
moins jusqu'à atteindre un score optimal (1). Le score optimal est atteint
en peu d'itérations.
'''

def tp2prog3():
	X,Y=make_classification(n_samples=100000,n_features=20,n_informative=15, \
	n_classes=3)
	X_train,X_test,Y_train,Y_test=\
	train_test_split(X,Y,test_size=0.3,random_state=random.seed())
	print ('train')
	i = 1
	while i < 20:
		clf=tree.DecisionTreeClassifier(max_leaf_nodes=500*i) #I
		i +=1
		clf = clf.fit(X_train,Y_train)
		print("%6.4f" %clf.score(X_train,Y_train))
	print ('test')
	i = 1
	while i < 20:
		clf=tree.DecisionTreeClassifier(max_leaf_nodes=500*i) #I
		i +=1
		clf = clf.fit(X_test,Y_test)
		print("%6.4f" %clf.score(X_test,Y_test))

# exercice 4
'''
Au début le classifieur n'est pas très précis.
A force de relancer avec une profondeur plus importante,
le classifieur est de plus en plus précis.
Il progresse d'ailleurs très vite sur les premières itérations et progresse
bien plus lentement dès lors qu'il se rapproche d'un score de 1.
'''

def tp2prog4():
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




# 4) INTERVALLE DE CONFIANCE POUR L'ERREUR ESTIMEE D'UN CLASSIFIEUR


def tp2prog5():
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
		
		f = 1-clf.score(X_2,Y_2)
		#print ("	f = %6.4f" %f)
		if f<binf or f>bsup:
			nbErr+=1
	print ("nb de fois ou f n'appartient pas a I: %d" %nbErr)




# 5) EXERCICE DE SYNTHESE


def get_leaf_with_crossed_validation(X, Y):
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
	#print(["{:4.2f}".format(s) for s in scores])
	k=scores.index(max(scores))+1
	print("meilleure valeur pour k : ", k)
	return k

def value_error (clf,X_train, Y_train, X_test, Y_test):
	N=1000
	e = 1-clf.score(X_test,Y_test)
	print("Erreur de %6.4f" %e)
	binf = e - 1.96 * math.sqrt((e * (1-e))/N)
	bsup = e + 1.96 * math.sqrt((e * (1-e))/N)
	
	print ("intervalle de confiance I")
	print ("	binf = %6.4f" %binf)
	print ("	bsup = %6.4f" %bsup)

def train_clf (crit,nbLeaf,X_train,Y_train,X_test,Y_test):
	
	clf=tree.DecisionTreeClassifier(criterion=crit, max_leaf_nodes=nbLeaf)
	clf = clf.fit(X_train,Y_train)
	value_error(clf,X_train, Y_train, X_test, Y_test)
	
	return clf

def mcNemar (Y_test, Y_predG, Y_predE):
	n10=0
	n01=0
	
	for test in Y_test:
		if Y_test[test] == Y_predG[test]:		
			n10+=1
		if Y_test[test] != Y_predE[test]:
			n01+=1
	
	res = pow(abs(n01-n10) - 1, 2) / (n01+n10)
	print ("mcNemar : ", res)
	

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
	print()
	
	# Gini
	print ('Gini')
	clfG = train_clf('gini', L_train, X_train,Y_train,X_test,Y_test)
	print()
	print ('entropy')
	clfE = train_clf('entropy', L_train, X_train,Y_train,X_test,Y_test)
	print()
	
	Y_predG = clfG.predict(X_test)
	Y_predE = clfE.predict(X_test)
	
	cmG = confusion_matrix(Y_test, Y_predG)
	cmE = confusion_matrix(Y_test, Y_predE)
	print('confusionMatrixG : ')
	print(cmG)
	print()
	print('confusionMatrixE : ')
	print(cmE)
	print()
	
	mcNemar(Y_test, Y_predG, Y_predE)



#tp2prog1()
#tp2prog2()
#tp2prog3()
#tp2prog4()
#tp2prog5()
synthese()













