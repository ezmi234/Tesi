from interpretableai import iai
import pandas as pd

df = pd.read_csv("data_banknote_authentication.txt", header=None,
                 names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]
(train_X, train_y), (test_X, test_y) = iai.split_data('classification', X, y,
                                                      seed=1)

cart = iai.GridSearch(
    iai.OptimalTreeClassifier(
        random_seed=1,
        criterion='gini',
    ),
    max_depth=range(1, 6),
)
cart.fit(train_X, train_y)
cart.get_learner()

# print the decision tree
print(cart.get_learner())

oct = iai.GridSearch(
    iai.OptimalTreeClassifier(
        random_seed=1,
    ),
    max_depth=range(1, 6),
)
oct.fit(train_X, train_y)
oct.get_learner()

# print the decision tree
print(oct.get_learner())
