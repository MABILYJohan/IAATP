import numpy as np
import random 
from sklearn.neural_network import MLPClassifier

def gen1(nbEx, noise=0):
	""" g ́en ́eration artificielle de donn ́ees binaires dans 2 carr ́es imbriqu ́es"""	
	X = np.random.rand(nbEx,2)-0.5
	y = [max(X[i])<= 0.35 for i in range(nbEx)]
	for i in range(nbEx):
		if random.random() < noise:
			y[i] = 1 - y[i]
	return X,y

noise = 0.0

X,y = gen1(100, noise)
X_test,y_test = gen1(1000, noise)

clf = MLPClassifier(hidden_layer_sizes=(3), alpha = 0.001, \
		solver = 'lbfgs',activation = 'logistic')

clf.fit(X,y)
print(clf.score(X_test,y_test))

X = X*1000
X_test = X_test*1000
# les coordonn ́ees de chaque exemple sont multipli ́ees par 1000
# cela ne devrait rien changer pour l’apprentissage

clf.fit(X,y)
print(clf.score(X_test,y_test))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

clf.fit(X,y)
print(clf.score(X_test,y_test))
