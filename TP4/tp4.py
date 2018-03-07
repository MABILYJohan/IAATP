# Dan LE & Johan MABILY

from pylab import rand
import numpy as np
import matplotlib.pyplot as plt

nbData = 200

def generateData(n):
	"""
	generates a 2D linearly separable dataset with 2n samples.
	"""
	X = (2*np.random.rand(2*n,2)-1)/2 - 0.5
	X[:n,1] += 1
	X[n:,0] += 1
	Y = np.ones([2*n,1])
	Y[n:] -= 2
	return X,Y
	

def visual(X, Y):
	point=0
	while point < len(X):
		if Y[point] == 1:
			plt.plot(X[point][0], X[point][1], 'ob')
		else:
			plt.plot(X[point][0], X[point][1], 'or')
		point+=1
	"""plt.legend()
	plt.show()"""


def perceptron (X, Y):
	w = []
	w = np.zeros(len(X[0]))
	print (w)
	
	flag=1
	while flag:
		flag=0
		for i in range(0, len(Y)):
			if Y[i] * np.vdot(w, X[i]) <= 0:
				w = w + Y[i]*X[i]
				print(w)
				flag=1
	return w
				
			
def exo2(w):
	plt.plot([-1,1],[w[0]/w[1],-w[0]/w[1]])
	plt.legend()
	plt.show()


def exo ():
	X_learn, Y_learn = generateData(nbData)
	print (X_learn)
	print()
	print (Y_learn)
	print ()
	visual (X_learn, Y_learn)
	
	w = perceptron(X_learn, Y_learn)
	exo2(w)
	
exo()
