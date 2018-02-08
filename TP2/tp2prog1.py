from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree

'''
def ex_1 ():
	X,Y=make_classification(n_samples=200,n_features=2,n_redundant=0,\
	n_clusters_per_class=1,n_classes=3)
	import pylab as pl
	pl.scatter(X[:,0],X[:,1],c=Y)
	pl.show()
	#pl.scatter(X[:,0],X[:,2],c=Y)
	#pl.show()


def ex_2 ():
	
	#help(tree.DecisionTreeClassifier)
	
	clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3, max_leaf_nodes=5)
	clf=clf.fit(iris.data, iris.target)
	print(clf.predict([iris.data[50,:]]))
	print(clf.score(iris.data,iris.target))
	tree.export_graphviz(clf,out_file="lala.dot")

ex_2()
'''

def ex1():
	clf=tree.DecisionTreeClassifier(max_leaf_nodes=3)
	print (iris.data)
	print (iris.target)
	clf=clf.fit(iris.data, iris.target)
	tree.export_graphviz(clf,out_file="ex1-1.dot")

ex1()
