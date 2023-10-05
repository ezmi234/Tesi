from interpretableai import iai
import pandas as pd
import matplotlib.pyplot as plt

names = ['En_count','O_Quant','R_Times','OTimes','OTimes_S','O_monthes','class']

df = pd.read_csv("../data/inliners/inliers2.csv",skiprows=1, usecols=range(1, 8), header=None,
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