from interpretableai import iai
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque


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


names = ['En_count', 'O_Quant', 'R_Times', 'OTimes', 'OTimes_S', 'O_monthes', 'class']

df = pd.read_csv("../data/inliners/inliers2.csv", skiprows=1, usecols=range(1, 8), header=None,
                 names=names)

X = df.iloc[:, 0:6]
y = df.iloc[:, 6]

(train_X, train_y), (test_X, test_y) = iai.split_data('classification', X, y,
                                                      seed=1)

grid = iai.GridSearch(
    iai.OptimalTreeClassifier(
        random_seed=1,
        criterion='gini',
    ),
    max_depth=25,
)

grid.fit(train_X, train_y)
lnr = grid.get_learner()

print(lnr.get_num_nodes(), 'nodes')
print(lnr)
bfs_tree(lnr)
