import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from scipy import stats

df = pd.read_csv("data_banknote_authentication.txt", header=None,
                    names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=1, random_state=1)

# gini index
clf = DecisionTreeClassifier()

# gini index random
# clf = DecisionTreeClassifier(splitter='random', max_depth=5)

# entropy min
# clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)

# entropy random
# clf = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=5)

clf.fit(X_train, y_train)

tree_rules = tree.export_text(clf, feature_names=list(X.columns))
print(tree_rules)
# y_pred = clf.predict(X_test)
#
# print("Accuracy:", accuracy_score(y_test, y_pred))
#
# # Calcolo dell'altezza massima dell'albero
altezza_massima = clf.get_depth()
#
# # Calcolo del numero totale di nodi
numero_nodi = clf.tree_.node_count
#
# # Calcolo dell'outdegree medio
# outdegree_medio = numero_nodi / (numero_nodi - 1)  # In un albero, ogni nodo eccetto la radice ha un solo padre
#
# # Calcolo della profondità delle foglie
# profondita_foglie = clf.tree_.max_depth
#
# # Calcolo della precisione sul set di test
# accuratezza = clf.score(X_test, y_test)
#
# # Stampa delle informazioni
print(f'Altezza massima: {altezza_massima}')
print(f'Numero totale di nodi: {numero_nodi}')
# print(f'Outdegree medio: {outdegree_medio}')
# print(f'Profondità delle foglie: {profondita_foglie}')
# print(f'Accuratezza sul set di test: {accuratezza}')
#
# # figure fontsize 8
# plt.figure(figsize=(20, 12))
# tree.plot_tree(clf, feature_names=['variance', 'skewness', 'curtosis', 'entropy'], class_names=['0', '1'], fontsize=8)
# plt.show()




