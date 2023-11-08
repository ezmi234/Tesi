from interpretableai import iai
import pandas as pd
# import matplotlib.pyplot as plt

df = pd.read_csv("../data/banknote-authentication/banknote-authentication.txt", header=None,
                 names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]
(train_X, train_y), (test_X, test_y) = iai.split_data('classification', X, y,
                                                      seed=1)

grid = iai.GridSearch(
    iai.OptimalTreeClassifier(
        random_seed=1,
        # criterion='gini',
    ),
    max_depth=5,
)
grid.fit(train_X, train_y)
lnr = grid.get_learner()

print(grid.get_grid_result_details())
print(grid.get_grid_result_summary())

# print the type of the learner
print(type(lnr))

# print the accuracy on the test set
print(lnr.score(test_X, test_y))

# print the accuracy on the training set
print(lnr.score(train_X, train_y))

# print the number of nodes in the tree
print(lnr.get_num_nodes(), 'nodes')

# print the depth of the tree
print(lnr.get_depth(20), 'depth')




# plot the decision tree

# plot = cart.get_learner().TreePlot()
# plot.show_in_browser()

# oct = iai.GridSearch(
#     iai.OptimalTreeClassifier(
#         random_seed=1,
#     ),
#     max_depth=range(1, 6),
# )
# oct.fit(train_X, train_y)
# oct.get_learner()
#
# # print the decision tree
# print(oct.get_learner())
