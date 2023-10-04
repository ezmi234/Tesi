from interpretableai import iai
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data_banknote_authentication.txt", header=None,
                 names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]
(train_X, train_y), (test_X, test_y) = iai.split_data('classification', X, y,
                                                      seed=1)

cart = iai.GridSearch(
    iai.OptimalTreeClassifier(
        random_seed=1,
        # criterion='gini',
    ),
    max_depth=5,
)
cart.fit(train_X, train_y)
cart.get_learner()

# plot the decision tree

plot = cart.get_learner().TreePlot()
plot.show_in_browser()

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
