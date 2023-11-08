from interpretableai import iai
import pandas as pd
# import matplotlib.pyplot as plt
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


def bfs_tree(tree):
    # Initialize a queue for BFS
    queue = deque([(1, 0)])  # (node_index, depth)

    while queue:
        node_index, depth = queue.popleft()

        if tree.is_leaf(node_index):
            print(f"At depth {depth}: Leaf node with class {tree.get_num_samples(node_index)}")
            # print(f"Class probabilities: {tree.get_classification_proba(node_index)}")
        else:
            print(f"At depth {depth}: Split on feature {tree.get_split_feature(node_index)},"
                  f" threshold {tree.get_split_threshold(node_index)}")
            queue.append((tree.get_lower_child(node_index), depth + 1))
            queue.append((tree.get_upper_child(node_index), depth + 1))

def convert_tree_to_json(tree, node_index=1, depth=0):
    json_tree = {}
    if tree.is_leaf(node_index):
        json_tree["type"] = "leaf"
        json_tree["num_samples"] = tree.get_num_samples(node_index)
        json_tree["class_probabilities"] = tree.get_classification_proba(node_index)
        json_tree["label"] = tree.get_classification_label(node_index)
        json_tree["depth"] = depth
        json_tree["node_index"] = node_index-1
    else:
        json_tree["type"] = "split"
        json_tree["split_feature"] = tree.get_split_feature(node_index)
        json_tree["split_threshold"] = tree.get_split_threshold(node_index)
        json_tree["depth"] = depth
        json_tree["node_index"] = node_index-1
        json_tree["left_child"] = convert_tree_to_json(tree, tree.get_lower_child(node_index), depth + 1)
        json_tree["right_child"] = convert_tree_to_json(tree, tree.get_upper_child(node_index), depth + 1)

    return json_tree


names = ['CF/TD', 'NI/TA', 'CA/CL', 'CA/NS', 'class']

df = pd.read_csv("../data/nath_jones/nath_jones.csv", skiprows=1, usecols=range(1, 6), header=None,
                 names=names)

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

(train_X, train_y), (test_X, test_y) = iai.split_data('classification', X, y, train_proportion=0.99,
                                                      seed=42)
print(len(train_X), len(train_y), len(test_X), len(test_y))
grid = iai.GridSearch(
    iai.OptimalTreeClassifier(
        # criterion='gini',
        random_seed=1,
    ),
    max_depth=4,
)
tick = time.time()
grid.fit(train_X, train_y)
tock = time.time()
print(tock-tick, 'seconds')
lnr = grid.get_learner()

save_tree(lnr, 'decision_tree.pkl')

# print(grid.get_grid_result_details())
# print(grid.get_grid_result_summary())
#
# # print the type of the learner
# print(type(lnr))
#
# # print the accuracy on the test set
# print(lnr.score(test_X, test_y))
# print(lnr.score(train_X, train_y))
#
# # print the accuracy on the training set
# print(lnr.score(train_X, train_y))
#
# # print the number of nodes in the tree
# print(lnr.get_num_nodes(), 'nodes')
#
# # print the depth of the tree
# print(lnr.get_depth(20), 'depth')
#
# print(lnr.get_parent(20), 'parent')
# print(lnr.get_lower_child(20), 'lower child')
# print(lnr.get_upper_child(20), 'upper child')
# print(lnr.is_leaf(20), 'leaf')

# plot the decision tree

plot = grid.get_learner().TreePlot()
plot.show_in_browser()

json_tree = convert_tree_to_json(lnr)
with open('decision_tree.json', 'w') as file:
    json.dump(json_tree, file, indent=4)


