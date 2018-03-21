
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_diabetes

Z = load_diabetes().target#[x/25-1 for x in range(51)]
X = load_diabetes().data#[[x] for x in Z]

def f1(x):
	return x**2
def f2(x):
	return math.sin(x)
def f3(x):
	return abs(x)
def f4(x):
	return max(min(math.ceil(x),1),0)

l_f = [f1, f2, f3, f4]

for f in l_f:
	y = [f(x) for x in Z]
	
	clf = MLPRegressor(hidden_layer_sizes=(3,),solver='lbfgs',activation='tanh',\
	learning_rate ='adaptive')
	clf.fit(X, y)
	yy = clf.predict(X)
	plt.scatter(Z,y)
	plt.plot(Z,yy, label = 'Cible')
	plt.legend()
	plt.show()
