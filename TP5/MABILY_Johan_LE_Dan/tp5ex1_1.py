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

noise = 10.0

X,y = gen1(100, noise)

clf = MLPClassifier(hidden_layer_sizes=(3), alpha = 0.001, \
		solver = 'lbfgs',activation = 'logistic')

clf.fit(X,y)

#print(clf.coefs_, clf.intercepts_)
print(clf.coefs_)

X_test,y_test = gen1(1000, noise)

print(clf.score(X_test,y_test))

gen1(2,noise)
