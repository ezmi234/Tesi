import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from collections import deque
import pickle
import json
import time

def save_tree(tree, filename):
    with open(filename, 'wb') as file:
        pickle.dump(tree, file)

def load_tree(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
def bfs_tree(tree, X):
    # Initialize a queue for BFS
    queue = deque([(0, 0)])  # (node_index, depth)

    while queue:
        node_index, depth = queue.popleft()

        if tree.children_left[node_index] != tree.children_right[node_index]:
            print(
                f"At depth {depth}: Split on feature {X.columns[tree.feature[node_index]]}, threshold {tree.threshold[node_index]}")
            queue.append((tree.children_left[node_index], depth + 1))
            queue.append((tree.children_right[node_index], depth + 1))
        else:
            print(f"At depth {depth}: Leaf node with class {tree.value[node_index]}")


def convert_tree_to_json(tree, node_index=0, depth=0):
    json_tree = {}
    feature_names = ['CF/TD', 'NI/TA', 'CA/CL', 'CA/NS']
    if tree.children_left[node_index] == tree.children_right[node_index]:
        json_tree["type"] = "leaf"
        json_tree["num_samples"] = int(tree.n_node_samples[node_index])
        json_tree["class_probabilities"] = tree.value[node_index].tolist()[0]
        json_tree["label"] = int(tree.value[node_index].argmax())
        json_tree["depth"] = int(depth)
        json_tree["node_index"] = int(node_index)
    else:
        json_tree["type"] = "split"
        json_tree["split_feature"] = feature_names[int(tree.feature[node_index])]
        json_tree["split_threshold"] = float(tree.threshold[node_index])
        json_tree["depth"] = int(depth)
        json_tree["node_index"] = int(node_index)
        json_tree["left_child"] = convert_tree_to_json(tree, tree.children_left[node_index], depth + 1)
        json_tree["right_child"] = convert_tree_to_json(tree, tree.children_right[node_index], depth + 1)

    return json_tree

names = ['CF/TD', 'NI/TA', 'CA/CL', 'CA/NS', 'class']

df = pd.read_csv("../data/nath_jones/nath_jones.csv", skiprows=1, usecols=range(1, 6), header=None,
                 names=names)

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
# print(len(X_train), len(y_train), len(X_test), len(y_test))

# gini index
clf = DecisionTreeClassifier()

tick = time.time()

clf.fit(X_train, y_train)

tock = time.time()
print("Time: ", tock - tick)
#
save_tree(clf, 'decision_tree.pkl')
# bfs_tree(clf.tree_, X_train)
#
# tree_rules = tree.export_text(clf, feature_names=list(X.columns))
# print(tree_rules)


# plot the decision tree
# plt.figure(figsize=(10, 8))
# tree.plot_tree(clf, feature_names=names, class_names=['0', '1'], fontsize=8)
# plt.show()
#
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# #
# # # Calcolo dell'altezza massima dell'albero
# altezza_massima = clf.get_depth()
# #
# # # Calcolo del numero totale di nodi
# numero_nodi = clf.tree_.node_count
# #
# # # Stampa delle informazioni
# print(f'Altezza massima: {altezza_massima}')
# print(f'Numero totale di nodi: {numero_nodi}')

# # Calcolo dell'importanza delle feature
importanza_feature = clf.feature_importances_
# #
# # # Stampa dell'importanza delle feature
print(f'Importanza delle feature: {importanza_feature}')

json_tree = convert_tree_to_json(clf.tree_)
print(json_tree)
print(type(json_tree))
with open('decision_tree.json', 'w') as file:
    json.dump(json_tree, file, indent=4)