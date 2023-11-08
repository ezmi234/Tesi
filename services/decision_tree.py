import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from interpretableai import iai
from collections import deque

def split_data(X, y, train_ratio, seed, tree):
    if (tree == 'cart'):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=seed)
        return X_train, X_test, y_train, y_test
    elif (tree == 'oct'):
        (X_train, y_train), (X_test, y_test) = iai.split_data('classification', X, y, train_proportion=0.7,
                                                              seed=seed)
        return X_train, X_test, y_train, y_test
    else:
        return None

def train_cart(X, y, max_depth=4):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)
    return clf

def train_oct(X, y, max_depth=4, criterion="misclassification"):
    grid = iai.GridSearch(
        iai.OptimalTreeClassifier(
            criterion=criterion,
            random_seed=1,
        ),
        max_depth=range(1, max_depth + 1),
    )
    grid.fit(X, y)
    return grid.get_learner()


def predict(node_data, x):
    queue = deque()
    queue.append(node_data)
    while len(queue) != 0:
        node = queue.popleft()
        if node['type'] == 'split':
            if float(x[node['split_feature']]) <= node['split_threshold']:
                queue.append(node['left_child'])
            else:
                queue.append(node['right_child'])
        else:
            return node['label']


def test_tree_on_data(node_data, x, y):
    right = 0
    for i in range(len(x)):
        y_pred = predict(node_data, x[i])
        if y_pred == int(y[i]):
            right += 1

    return right / len(x)

def predictOct(node_data, x):
    queue = deque()
    queue.append(node_data)
    while len(queue) != 0:
        node = queue.popleft()
        if node['type'] == 'split':
            if float(x[node['split_feature']]) < node['split_threshold']:
                queue.append(node['left_child'])
            else:
                queue.append(node['right_child'])
        else:
            return node['label']


def test_tree_on_dataOct(node_data, x, y):
    right = 0
    for i in range(len(x)):
        y_pred = predictOct(node_data, x[i])
        if y_pred == int(y[i]):
            right += 1

    return right / len(x)

def getNumNodes(node_data):
    queue = deque()
    queue.append(node_data)
    numNodes = 0
    while len(queue) != 0:
        node = queue.popleft()
        numNodes += 1
        if node['type'] == 'split':
            queue.append(node['left_child'])
            queue.append(node['right_child'])
    return numNodes