from interpretableai import iai
import pandas as pd
import time

import dataset

datasets = ['balance-scale', 'breast-cancer', 'car-evaluation', 'hayes-roth', 'house-votes-84',
            'soybean-small', 'spect', 'tic-tac-toe', 'monks-1', 'monks-2', 'monks-3']

alpha = [0, 0.01, 0.1]
depth = [2, 3, 4, 5]
seeds = [37, 42, 53]

train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

res_iai = pd.DataFrame(columns=['instance', 'depth', 'seed', 'train_acc', 'val_acc', 'test_acc', 'train_time',
                                'num_nodes'])

for data in datasets:
    for d in depth:
        for s in seeds:
            x, y = dataset.loadData(data)
            (train_X, train_y), (test_X, test_y) = iai.split_data('classification', x, y,
                                                                  train_proportion=1 - train_ratio,
                                                                  seed=s)
            (val_x, val_y), (test_X, test_y) = iai.split_data('classification', test_X, test_y,
                                                              train_proportion=test_ratio / (test_ratio + val_ratio),
                                                              seed=s)
            grid = iai.GridSearch(
                iai.OptimalTreeClassifier(random_seed=s),
                max_depth=d
            )
            tick = time.time()
            grid.fit(train_X, train_y)
            tock = time.time()
            best_model = grid.get_learner()
            train_acc = best_model.score(train_X, train_y)
            val_acc = best_model.score(val_x, val_y)
            test_acc = best_model.score(test_X, test_y)
            num_nodes = best_model.get_num_nodes()

            row = {'instance': data, 'depth': d, 'seed': s, 'train_acc': train_acc, 'val_acc': val_acc,
                   'test_acc': test_acc, 'train_time': tock - tick, 'num_nodes': num_nodes}
            res_iai = pd.concat([res_iai, pd.DataFrame([row])], ignore_index=True)

res_iai.to_csv('./res/iai.csv', index=False)
