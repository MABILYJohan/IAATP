import numpy as np
import matplotlib.pyplot as plt

# commandes utiles
X = np.zeros([5,3]) # tableau de 0 de dimension 5x3
Y = np.ones([3,2]) # tableau de 1 de dimension 5x3
v = np.ones(3) # vecteur contenant trois 1
X[1:4,:2] = Y # remplacement d’une partie du tableau X
X.shape # dimensions de X
np.random.rand(10) # 10 nombres al ́eatoires entre 0 et 1
Z = np.random.random([4,4]) # matrice al ́eatoire
np.random.normal(0,1,10) # 10 nombres al ́eatoires g ́en ́er ́es par la Gaussienne N(0,1)
np.dot(X,Y) # produit matriciel
np.dot(X,v) # produit de la matrice X et du vecteur v
np.transpose(X) # transpos ́ee de X
np.linalg.inv(Z) # inverse de Z





def GenData(x_min, x_max, w, nbEx, sigma):
	""" g ́en`ere al ́eatoirement n donn ́ees du type (x,w0 + <w_1:n,x> + e) o`u
	- w est un vecteur de dimension d + 1
	- x_min <= |x_i| <= x_max pour les d coordonn ́ees x_i de x
	- e est un bruit Gaussien de moyenne nulle et d’ ́ecart type sigma
	Retourne deux np.array de forme (nbEx,1)
	"""
	d = len(w) - 1
	X = (x_max-x_min)*np.random.rand(nbEx,d) + x_min
	Y = np.dot(X,w[1:]) + w[0] + np.random.normal(0,sigma,nbEx)
	Y = Y.reshape(nbEx,1)
	return X, Y


def AddOne(X):
	""" X est un tableau n x d ; retourne le tableau n x (d+1) consistant `a rajouter une colonne de 1 `a X """
	(n,d) = X.shape
	Xnew = np.zeros([n,d+1])
	Xnew[:,1:]=X
	Xnew[:,0]=np.ones(n)
	return Xnew

def RegLin(X,Y):
	""" X est un tableau n x d ; Y est un tableau de dimension n x 1
	retourne le vecteur w de dimension d+1, r ́esultat de la r ́egression lin ́eaire """
	Z = AddOne(X)
	return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Z),Z)),np.transpose(Z)),Y[:,0])


def RSS(X,Y,w):
	""" Residual Sum of Squares  """
	v = Y[:,0]- (np.dot(X,w[1:]) + w[0])
	return np.dot(v,v)


def ex1():
	x_min = -5
	x_max = 5
	nbEx = 10
	sigma = 1.0
	w = np.array([-1,2])
	X,Y = GenData(x_min, x_max, w,nbEx,sigma)
	#print(X.shape,Y.shape)
	w_estime = RegLin(X,Y)
	#print(RSS(X,Y,w),RSS(X,Y,w_estime))
	
	"""plt.scatter(X,Y)
	plt.plot([x_min,x_max],[w[1]*x_min + w[0],w[1]*x_max + w[0]], label = 'Cible')
	plt.plot([x_min,x_max],[w_estime[1]*x_min + w_estime[0],w_estime[1]*x_max + w_estime[0]], label = 'Estimation')
	plt.legend()
	plt.show()"""
	
	#ex2(x_min, x_max, nbEx, sigma)
	ex3_3(x_min, x_max)


	

'''
les w sont des estimateurs, si l’on repete un grand nombre d'experiences
avec le meme modèle, les moyennes des estimations convergent vers les paramètres
du modèle.
$ [-1  2 -1  3] [-1.04478019  1.97535373 -0.94623313  2.92427654]
$ [-1  2 -1  3] [-0.93264756  1.99860932 -0.99390471  3.00092662]

'''
def ex2(x_min, x_max, nbEx, sigma):
	w = np.array([-1,2,-1,3])
	w_mean = np.zeros(4)
	for i in range(10):
		X,Y = GenData(x_min, x_max, w, nbEx, sigma)
		w_mean += RegLin(X,Y)
	w_mean = w_mean/10
	#print(w,w_mean)
	nbEx = 1000
	X, Y = GenData(x_min, x_max, w, nbEx, sigma)
	w_estime = RegLin(X,Y)
	#print(w,w_estime)
	
def ex3_1():
	Z = np.loadtxt("./TP3.data1")
	plt.scatter(Z[:,0],Z[:,1])
	plt.show()


def poly (x, d):
	L=[]
	for i in range(1,d+1):
		L.append(pow(x,i))
	return L

def polyTab(X,d):
	tab=[]
	for i in range(0, len(X)):
		tab.append(poly(i, d))	
	return tab


def ex3_3():
	
	x_min = -5
	x_max = 5
	nbEx = 10
	sigma = 1.0
	Z = np.loadtxt("./TP3.data1")
	for d in range(1, 10+1):
		tab = polyTab(X, d)
		z = Z[:,1]
		z = z.array()
		z.reshape(len(z),1)
		w_estime = RegLin(polyTab(Z[:,0],Z[:,1]))
		abscisses = np.linspace(x_min, x_max, 50)
		ordonnees = [np.dot(poly(x), w) for x in abscisses]
		plt.scatter(Z[:,0],Z[:,1])
		plt.plot(abscisses,ordonnees)
		plt.legend()
		plt.show()
		#print (tab)
	
	'''	
	abscisses = np.linspace(x_min, x_max, 50)
	w = [1, 2, 3]
	ordonnees = [np.dot([1,x,x*x], w) for x in abscisses]
	
	plt.scatter(X,Y)
	plt.plot(abscisses,ordonnees)
	plt.legend()
	plt.show()
	'''


ex3_3()











