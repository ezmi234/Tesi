#!/usr/bin/env python

import time
from os import path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import dataset
from sklearn import tree

timelimit = 600
datasets = ['balance-scale', 'breast-cancer', 'car-evaluation', 'hayes-roth', 'house-votes-84',
            'soybean-small', 'spect', 'tic-tac-toe', 'monks-1', 'monks-2', 'monks-3']
alpha = [0, 0.01, 0.1]
depth = [2, 3, 4, 5]
seeds = [37, 42, 53]

train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

# create or load table
res_sk = pd.DataFrame(columns=['instance', 'depth', 'seed', 'train_acc', 'val_acc', 'test_acc', 'train_time'])
# if path.isfile('./res/oct.csv'):
#     res_oct = pd.read_csv('./res/oct.csv')
# else:
#     res_oct = pd.DataFrame(columns=['instance', 'depth', 'alpha', 'seed',
#                                     'train_acc', 'val_acc', 'test_acc', 'train_time', 'gap'])
# if path.isfile('./res/mfoct.csv'):
#     res_mfoct = pd.read_csv('./res/mfoct.csv')
# else:
#     res_mfoct = pd.DataFrame(columns=['instance', 'depth', 'alpha', 'seed',
#                                       'train_acc', 'val_acc', 'test_acc', 'train_time', 'gap'])
# if path.isfile('./res/boct.csv'):
#     res_boct = pd.read_csv('./res/boct.csv')
# else:
#     res_boct = pd.DataFrame(columns=['instance', 'depth', 'seed',
#                                      'train_acc', 'val_acc', 'test_acc', 'train_time', 'gap'])
# if path.isfile('./res/soct.csv'):
#     res_soct = pd.read_csv('./res/soct.csv')
# else:
#     res_soct = pd.DataFrame(columns=['instance', 'method', 'depth', 'alpha', 'seed',
#                                      'train_acc', 'val_acc', 'test_acc', 'train_time', 'gap'])

for data in datasets:
    for d in depth:
        for s in seeds:
            x, y = dataset.loadData(data)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio, random_state=s)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                            test_size=test_ratio / (test_ratio + val_ratio),
                                                            random_state=s)
            clf = tree.DecisionTreeClassifier(max_depth=d)
            tick = time.time()
            clf.fit(x_train, y_train)
            tock = time.time()
            train_time = tock - tick
            train_acc = accuracy_score(y_train, clf.predict(x_train))
            val_acc = accuracy_score(y_val, clf.predict(x_val))
            test_acc = accuracy_score(y_test, clf.predict(x_test))
            print(data, 'cart-d{}'.format(d), 'train acc:', train_acc, 'val acc:', val_acc)
            row = {'instance': data, 'depth': d, 'seed': s, 'train_acc': train_acc,
                   'val_acc': val_acc, 'test_acc': test_acc, 'train_time': train_time}
            res_sk = pd.concat([res_sk, pd.DataFrame([row])], ignore_index=True)

res_sk.to_csv('./res/sk.csv', index=False)
