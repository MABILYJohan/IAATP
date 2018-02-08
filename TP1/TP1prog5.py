import random
import numpy as np
import pylab as pl
from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

#N=5 #haut du tableau
#d=2 # dimension


def distance_entre_2_points (a, b, d):
	i=0
	coord = []
	while i<d:
		coord.append(a[i] + b[i])
		i+=1
	tmp = 0;
	i=0
	while i<d: 
		tmp = max(tmp, coord[i])
		i+=1
	return tmp


#calcule la moyenne des distances des points de X au centre
def distance_au_centre (X, d) :
	#print ('X')
	#print (X)
	somme=0 
	#print ('centre')
	centre = 0.5*np.ones(d)
	print (centre)
	for point in X:
		#print ('point')
		print (point)
		#somme= max (point, centre)
		i=0
		somme += distance_entre_2_points (point, centre, d)
	somme = somme/len(X)
	return somme


#Calcule la distance minimale d’un point de X au centre
def voisin_le_plus_proche_du_centre (X, d):
	mini=100000
	centre = 0.5*np.ones(d)
	for point in X:
		tmp = distance_entre_2_points (point, centre, d)
		if (mini > tmp): 
			mini = tmp
			finalPoint = point
	return finalPoint

'''
N = 5
d=2
tab = np.random.rand(N, d)
print(distance_au_centre(tab, d))
print("voisin le plus proche du centre")
print(voisin_le_plus_proche_du_centre(tab, d))
#print (0.5*np.ones(d))
#print (tab)
'''
'''
Cette fonction crée 20 fois 10 tableaux
chaque tranche de 10 tableaux sont de dimensions 1 à 21
Ces tableaux contiennent chacun 100 points de valeurs aléatoires
'''

def ex3_2 ():
	for d in range(1,21):
		dist = []
		v = []
		for i in range(10):
			X = np.random.rand(100,d)
			dist.append(distance_au_centre(X, d))
			v.append( voisin_le_plus_proche_du_centre(X, d))
		print(np.mean(dist), np.mean(v))

#ex3_2() 




def damier(dimension, grid_size, nb_examples, noise):
	data = np.random.rand(nb_examples,dimension)
	labels = np.ones(nb_examples)
	for i in range(nb_examples):
		x = data[i,:];
		for j in range(dimension):
			if int(np.floor(x[j]*grid_size)) % 2 != 0:
				labels[i]=labels[i]*(-1)
		if np.random.rand()<noise:
			labels[i]=labels[i]*(-1)
	return data, labels

'''
data, labels = damier (1, 10, 10, 0)
print ('data')
print (data)
print ('labels')
print (labels)
'''

def classifieur (X, Y):
	#from sklearn import neighbors
	#from sklearn.cross_validation import KFold
	#from sklearn.model_selection import KFold
	#kf=KFold(len(X),n_folds=10,shuffle=True)
	kf=KFold(n_splits=7,shuffle=True)
	scores=[]
	#X_train,X_test,Y_train,Y_test=\
	#train_test_split(X,Y,test_size=0.3,random_state=random.seed())
	for k in range(1,5):
		score=0
		clf = neighbors.KNeighborsClassifier(k)
		#for learn,test in kf:
		for learn,test in kf.split(X):
			X_train=X[learn]
			Y_train=Y[learn]
			clf.fit(X_train, Y_train)
			X_test=X[test]
			Y_test=Y[test]
			score = score + clf.score(X_test,Y_test)
		scores.append(score)
	#print(["{:4.2f}".format(s) for s in scores])
	#print("meilleure valeur pour k : ",scores.index(max(scores))+1)


'''
On se rend compte que si on lance le programme avec toutes les variations
de l'énoncé, le programme va mettre un temps exponentiel à terminer.
'''
def ex3_2_1 ():
	#X, Y = damier (1, 8, 1000, 0)
	#print(X)
	#print (Y)
	#classifieur(X, Y)
	
	for dim in range(2,3):
		for nb_cases in range(2,3):
			for nb_ex in range (1000, 1100):
				noise = 0
				for i in range (1,3):
					X, Y = damier (dim, nb_cases, nb_ex, noise)
					noise+=0.1
					classifieur (X, Y)
					X_train,X_test,Y_train,Y_test=\
					train_test_split(X,Y,test_size=0.3,random_state=random.seed())
					print(i)
					print(len(X_train))
					print(len(X_test))
					print(len(X_train[Y_train==1]))
					print(len(X_train[Y_train==-1]))
					print("-------------------------------------")

	



ex3_2_1 ()
	




















