# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import random # pour pouvoir utiliser un g ́en ́erateur de nombres al ́eatoires

from sklearn import neighbors
nb_voisins = 15

irisData=load_iris()
X=irisData.data
Y=irisData.target

X_train,X_test,Y_train,Y_test=\
train_test_split(X,Y,test_size=0.3,random_state=random.seed())
#help(train_test_split)
print(len(X_train))
print(len(X_test))
print(len(X_train[Y_train==0]))
print(len(X_train[Y_train==1]))
print(len(X_train[Y_train==2]))

print()
"""
print(Y_train)
print()
print(Y_train==0)
print()
print(Y_train==1) """

clf = neighbors.KNeighborsClassifier(nb_voisins)
clf.fit(X_train, Y_train)
#print(clf.predict(X_train)) # Predit une catégoriie selon les données entrées
"""
print(clf.predict_proba([[ 5.4,  3.2,  1.6,  0.4]]))
"""
print(clf.score(X_test,Y_test))
"""
Z = clf.predict(X)
print(X[Z!=Y])
"""


