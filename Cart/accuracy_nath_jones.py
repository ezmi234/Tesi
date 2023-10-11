import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from collections import deque
import pickle
import numpy as np

def load_tree(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def bfs_tree(tree, x):
    # Initialize a queue for BFS
    queue = deque([(0, 0)])  # (node_index, depth)

    while queue:
        node_index, depth = queue.popleft()

        if tree.children_left[node_index] != tree.children_right[node_index]:
            print(
                f"At depth {depth}: Split on feature {tree.feature[node_index]}, threshold {tree.threshold[node_index]}")
            if x.iloc[tree.feature[node_index]] <= tree.threshold[node_index]:
                queue.append((tree.children_left[node_index], depth + 1))
            else:
                queue.append((tree.children_right[node_index], depth + 1))
        else:
            print(f"At depth {depth}: Leaf node with class {tree.value[node_index]}")


def test_tree_on_Xtest(tree, X_test, y_test):

    cont = 0
    p1 = 0
    p2 = 0
    for index, sample in X_test.iterrows():
        print(f"Sample {index}:")
        bfs_tree(tree.tree_, sample)

        # Predict and print class probabilities
        probabilities = tree.predict_proba(sample.to_frame().T)
        print(f"Predicted Class Probabilities: {probabilities}")
        predicted_class = tree.predict(sample.to_frame().T)[0]
        print(f"Predicted Class: {predicted_class}")
        true_class = y_test.loc[index]
        print(f"True Class: {true_class}\n")
        if predicted_class == true_class:
            p1 += 1
        cont += 1
    print(p1/cont)
    print(cont)

    # calculate accuracy
    y_pred = tree.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))



names = ['CF/TD', 'NI/TA', 'CA/CL', 'CA/NS', 'class']

df = pd.read_csv("../data/nath_jones/nath_jones.csv", skiprows=1, usecols=range(1, 6), header=None,
                 names=names)

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

loaded_tree = load_tree('decision_tree.pkl')

test_tree_on_Xtest(loaded_tree, X_test, y_test)

