import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from collections import deque


def bfs_tree(tree, X):
    # Initialize a queue for BFS
    queue = deque([(0, 0)])  # (node_index, depth)

    while queue:
        node_index, depth = queue.popleft()

        if tree.children_left[node_index] != tree.children_right[node_index]:
            print(f"At depth {depth}: Split on feature {X.columns[tree.feature[node_index]]}, threshold {tree.threshold[node_index]}")
            queue.append((tree.children_left[node_index], depth + 1))
            queue.append((tree.children_right[node_index], depth + 1))
        else:
            print(f"At depth {depth}: Leaf node with class {tree.value[node_index]}")


names = ['En_count','O_Quant','R_Times','OTimes','OTimes_S','O_monthes','class']

df = pd.read_csv("../data/inliners/inliers2.csv",skiprows=1, usecols=range(1, 8), header=None,
                    names=names)

X = df.iloc[:, 0:6]
y = df.iloc[:, 6]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.5, random_state=1)

# gini index
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

bfs_tree(clf.tree_, X_train)

# tree_rules = tree.export_text(clf, feature_names=list(X.columns)
# print(tree_rules)

# plot the decision tree
plt.figure(figsize=(20, 12))
tree.plot_tree(clf, feature_names=names, fontsize=8)
plt.show()

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Calcolo dell'altezza massima dell'albero
altezza_massima = clf.get_depth()

# Calcolo del numero totale di nodi
numero_nodi = clf.tree_.node_count


# Stampa delle informazioni
print(f'Altezza massima: {altezza_massima}')
print(f'Numero totale di nodi: {numero_nodi}')
