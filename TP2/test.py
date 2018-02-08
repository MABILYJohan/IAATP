from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree

def ex_1 ():
	X,Y=make_classification(n_samples=100,n_features=2,n_redundant=0,\
	n_clusters_per_class=1,n_classes=2)
	import pylab as pl
	pl.scatter(X[:,0],X[:,1],c=Y)
	pl.show()
	pl.scatter(X[:,0],X[:,2],c=Y)
	pl.show()


def ex_2 ():
	
	#help(tree.DecisionTreeClassifier)
	
	clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3, max_leaf_nodes=5)
	clf=clf.fit(iris.data, iris.target)
	print(clf.predict([iris.data[50,:]]))
	print(clf.score(iris.data,iris.target))
	tree.export_graphviz(clf,out_file="test.dot")

ex_2()
