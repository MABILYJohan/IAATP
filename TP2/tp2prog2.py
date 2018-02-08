from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree

def ex2():
	clf=tree.DecisionTreeClassifier()
	clf=clf.fit(iris.data, iris.target)
	tree.export_graphviz(clf,out_file="tree_gini")
	clf=tree.DecisionTreeClassifier(criterion='entropy')
	clf=clf.fit(iris.data, iris.target)
	tree.export_graphviz(clf,out_file="tree_entropy")
ex2()
