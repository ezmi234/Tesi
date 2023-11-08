from interpretableai import iai
import pandas as pd
import matplotlib.pyplot as plt
import time
import dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from services import decision_tree
from os import path
from collections import deque

def bfs_tree(tree, X):
    # Initialize a queue for BFS
    queue = deque([(0, 0)])  # (node_index, depth)

    while queue:
        node_index, depth = queue.popleft()

        if tree.children_left[node_index] != tree.children_right[node_index]:
            print(f"At depth {depth}: Split on feature {X.columns[tree.feature[node_index]]}, threshold {tree.threshold[node_index]}")
            print(tree.impurity[node_index])
            queue.append((tree.children_left[node_index], depth + 1))
            queue.append((tree.children_right[node_index], depth + 1))
        else:
            print(f"At depth {depth}: Leaf node with class {tree.value[node_index]}")
            print(tree.n_node_samples[node_index], 'qui')
            print(tree.value[node_index], 'qui2')
            print()
            print()
            print(tree.value, 'qui3')


def process_node(node):
    if tree.children_left[node.children_left] == tree.children_right[node.children_right]:
        # This is a leaf node
        num_samples = node.n_node_samples
        class_distribution = node.value
        # Save or evaluate as needed
    else:
        # This is a non-leaf node
        impurity = node.impurity
        # Save or evaluate as needed


# datasets = ['balance-scale', 'breast-cancer', 'car-evaluation', 'hayes-roth', 'house-votes-84',
#             'soybean-small', 'spect', 'tic-tac-toe', 'monks-1', 'monks-2', 'monks-3']

# datasets = ['nath-jones', 'inliers', 'banknote-authentication']
datasets = ['car-evaluation',]
max_depth = 50
depth=[4]
s=42
overwrite = False

train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

for data in datasets:
    x, y = dataset.loadData(data)
    for d in depth:
        print('\n\nCART depth', d, 'on', data)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=s)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=test_ratio / (test_ratio + val_ratio), random_state=s)
        clf = tree.DecisionTreeClassifier(max_depth=d)

        start = time.time()
        clf = clf.fit(x_train, y_train)
        end = time.time()
        print('Training time', end - start)
        print('Training accuracy', accuracy_score(y_train, clf.predict(x_train)))
        n_nodes = clf.tree_.node_count
        print('Number of nodes', n_nodes)
        print('Number of leaves', clf.get_n_leaves())

        print('\n\nOCT depth', d, 'on', data)

        (train_X, train_y), (test_X, test_y) = iai.split_data('classification', x, y, train_proportion=train_ratio,
                                                              seed=1)

        grid = iai.GridSearch(
            iai.OptimalTreeClassifier(
                # criterion='gini',
                random_seed=1,
            ),
            max_depth=range(1, d + 1),
        )
        start = time.time()
        grid.fit(train_X, train_y)
        end = time.time()
        print('Training time', end - start)
        print('Training accuracy', accuracy_score(train_y, grid.predict(train_X)))
        print('Number of nodes', grid.get_learner().get_num_nodes())

        # plot = grid.get_learner().TreePlot()
        # plot.show_in_browser()
        #
        #
        # print("plot tree")
        # tree_rules = tree.export_text(clf)
        # with open("tree_rules"+data+".txt", "w") as text_file:
        #     text_file.write(tree_rules)

# res_sk = pd.DataFrame(columns=['instance', 'depth', 'seed', 'train_acc', 'val_acc', 'test_acc', 'train_time'])
#
# for data in datasets:
#     x, y = dataset.loadData(data)
#     x_train, x_test, y_train, y_test = decision_tree.split_data(x, y, train_ratio,
#                                                                 42, 'cart')
#     if (path.isfile('./DecisionTree/' + data + 'Cart' + '.pkl') and not overwrite):
#         cart = load_tree('./DecisionTree/' + data + 'Cart' + '.pkl')
#         print(data, 'cart loaded')
#     else:
#         cart = decision_tree.train_cart(x_train, y_train)
#         save_tree(cart, './DecisionTree/' + data + 'Cart' + '.pkl')
#         print(data, 'cart trained')
#
#     bfs_tree(cart.tree_, x_train)
#     print(data, 'cart accuracy:', accuracy_score(y_test, cart.predict(x_test)))
#
#     # plot the decision tree
#     plt.figure(figsize=(40, 20))
#     tree.plot_tree(cart, feature_names=x_train.columns, fontsize=8)
#     plt.show()
#
#     # x_train, x_test, y_train, y_test = decision_tree.split_data(x, y, train_ratio,
#     #                                                             42, 'oct')
#     # if (path.isfile('./DecisionTree/' + data + 'Oct' + '.pkl') and not overwrite):
#     #     oct = load_tree('./DecisionTree/' + data + 'Oct' + '.pkl')
#     #     print(data, 'oct loaded')
#     # else:
#     #     oct = decision_tree.train_oct(x_train, y_train, max_depth)
#     #     save_tree(oct, './DecisionTree/' + data + 'Oct' + '.pkl')
#     #     print(data, 'oct trained')
#     #
#     # print(data, 'oct accuracy:', oct.score(x_test, y_test, criterion='misclassification'))
#
