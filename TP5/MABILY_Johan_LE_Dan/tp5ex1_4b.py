import numpy as np
import random 
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_diabetes

def gen1(nbEx, noise=0):
	""" g ́en ́eration artificielle de donn ́ees binaires dans 2 carr ́es imbriqu ́es"""	
	X = np.random.rand(nbEx,2)-0.5
	y = [max(X[i])<= 0.35 for i in range(nbEx)]
	for i in range(nbEx):
		if random.random() < noise:
			y[i] = 1 - y[i]
	return X,y

noise = 0.0

X,y = load_diabetes().data, load_diabetes().target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=random.seed())


clf = MLPClassifier(hidden_layer_sizes=(2,), solver = 'lbfgs',activation = 'logistic')
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
print(clf.predict_proba(X_test))
# la fonction \texttt{softmax} est utilis ́ee pour estimer la
# probabilit ́e de chaque classe
