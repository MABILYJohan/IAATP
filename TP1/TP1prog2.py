# -*- coding: utf-8 -*-
import pylab as pl
from sklearn.datasets import load_iris
from sklearn import neighbors
nb_voisins = 15
irisData=load_iris()
X=irisData.data
Y=irisData.target
#help(neighbors.KNeighborsClassifier)
clf = neighbors.KNeighborsClassifier(nb_voisins)

#help(clf.fit)
clf.fit(X, Y)
#help(clf.predict)
print(clf.predict([[ 5.4,  3.2,  1.6,  0.4]]))
print(clf.predict_proba([[ 5.4,  3.2,  1.6,  0.4]]))
print(clf.score(X,Y))
Z = clf.predict(X)
print(X[Z!=Y])
