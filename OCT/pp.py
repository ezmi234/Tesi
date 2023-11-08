from interpretableai import iai
import pandas as pd
# import matplotlib.pyplot as plt
from collections import deque
import pickle


def load_tree(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def bfs_tree(tree, x):
    # Initialize a queue for BFS
    queue = deque([(1, 0)])  # (node_index, depth)

    while queue:
        node_index, depth = queue.popleft()

    if tree.get_lower_child(node_index) != tree.get_upper_child(node_index):
        print(f"At depth {depth}: Split on feature {tree.get_split_feature(node_index)},"
              f" threshold {tree.get_split_threshold(node_index)}")
        if x.iloc[tree.get_split_categories(node_index)] <= tree.get_split_threshold(node_index):
            queue.append((tree.get_lower_child(node_index), depth + 1))
        else:
            queue.append((tree.get_upper_child(node_index), depth + 1))
    else:
        print(f"At depth {depth}: Leaf node with class {tree.get_num_samples(node_index)}")


def test_tree_on_Xtest(tree, X_test, y_test):

    cont = 0
    p1 = 0
    p2 = 0
    for index, sample in X_test.iterrows():
        print(f"Sample {index}:")
        # bfs_tree(tree, sample)

        # Predict and print class probabilities
        probabilities = tree.predict_proba(sample.to_frame().T)
        print(f"Predicted Class Probabilities: {probabilities}")
        predicted_class = tree.predict(sample.to_frame().T)[0]
        print(f"Predicted Class: {predicted_class}")
        true_class = y_test[index]
        print(f"True Class: {true_class}\n")
        if predicted_class == true_class:
            p1 += 1
        cont += 1
    print(p1/cont)
    print(cont)

    # calculate accuracy
    print(tree.score(test_X, test_y, criterion='auc'))
    print(tree.score(test_X, test_y, criterion='misclassification'))


names = ['CF/TD', 'NI/TA', 'CA/CL', 'CA/NS', 'class']

df = pd.read_csv("../data/nath_jones/nath_jones.csv", skiprows=1, usecols=range(1, 6), header=None,
                    names=names)

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

(train_X, train_y), (test_X, test_y) = iai.split_data('classification', X, y, train_proportion=0.5,
                                                        seed=1)

lnr = load_tree('decision_tree.pkl')

test_tree_on_Xtest(lnr, test_X, test_y)