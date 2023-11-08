from services.datastructures import *
from interpretableai import iai
import pandas as pd
import time
import dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from services import decision_tree
from os import path
import json

datasets = ['nath-jones']

max_depth = 50
overwrite = True
train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

for data in datasets:
    x, y = dataset.loadData(data)
    x_train, x_test, y_train, y_test = decision_tree.split_data(x, y, train_ratio,
                                                                42, 'cart')
    if (path.isfile('./DecisionTree/' + data + 'Cart' + '.pkl') and not overwrite):
        # cart = load_tree('./DecisionTree/' + data + 'Cart' + '.pkl')
        print(data, 'cart loaded')
    else:
        cart = decision_tree.train_cart(x_train, y_train)
        print(cart)
        cart_json = convert_cart_to_json(cart.tree_)
        print(cart_json)
        with open('./DecisionTree/' + data + 'Cart' + '.json', 'w') as file:
            json.dump(cart_json, file)
        print(data, 'cart trained')