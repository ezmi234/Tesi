from services.datastructures import *
from interpretableai import iai
import pandas as pd
import time
import dataset
from sklearn.model_selection import train_test_split
from services import decision_tree
from os import path

# Datasets
datasets = ['nath-jones']

# Train and test variables
max_depth = 4
overwrite = False
train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

for data in datasets:
    x, y = dataset.loadData(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=test_ratio / (test_ratio + val_ratio),
                                                    random_state=42)

    if (path.isfile('./DecisionTree/' + data + 'Cart' + '.json') and not overwrite):
        with open('./DecisionTree/' + data + 'Cart' + '.json', 'r') as file:
            cart_json = json.load(file)
    else:
        cart = decision_tree.train_cart(x_train, y_train, max_depth)
        cart_json = convert_cart_to_json(cart.tree_)
        with open('./DecisionTree/' + data + 'Cart' + '.json', 'w') as file:
            json.dump(cart_json, file)

    print('Accuracy Test:',
          decision_tree.test_tree_on_data(cart_json, x_test.to_numpy().tolist(), y_test.to_numpy().tolist()))
    print('Accuracy Val:',
          decision_tree.test_tree_on_data(cart_json, x_val.to_numpy().tolist(), y_val.to_numpy().tolist()))
    print('Accuracy Train:',
          decision_tree.test_tree_on_data(cart_json, x_train.to_numpy().tolist(), y_train.to_numpy().tolist()))

    (train_X, train_y), (test_X, test_y) = iai.split_data('classification', x, y,
                                                          train_proportion=train_ratio,
                                                          seed=42)
    (val_x, val_y), (test_X, test_y) = iai.split_data('classification', test_X, test_y,
                                                      train_proportion=1 - test_ratio / (test_ratio + val_ratio),
                                                      seed=42)

    if (path.isfile('./DecisionTree/' + data + 'Oct' + '.json') and not overwrite):
        with open('./DecisionTree/' + data + 'Oct' + '.json', 'r') as file:
            oct_json = json.load(file)
    else:
        oct = decision_tree.train_oct(train_X, train_y, max_depth)
        # plot = oct.TreePlot()
        # plot.show_in_browser()
        # train_acc = oct.score(train_X, train_y)
        # val_acc = oct.score(val_x, val_y)
        # test_acc = oct.score(test_X, test_y)
        # print('Train accuracy', train_acc)
        # print('Val accuracy', val_acc)
        # print('Test accuracy', test_acc)
        oct_json = convert_oct_to_json(oct)
        with open('./DecisionTree/' + data + 'Oct' + '.json', 'w') as file:
            json.dump(oct_json, file)

    print('Accuracy Test:',
          decision_tree.test_tree_on_dataOct(oct_json, test_X.to_records(index=False).tolist(), test_y.tolist()))
    print('Accuracy Val:',
            decision_tree.test_tree_on_dataOct(oct_json, val_x.to_records(index=False).tolist(), val_y.tolist()))
    print('Accuracy Train:',
            decision_tree.test_tree_on_dataOct(oct_json, train_X.to_records(index=False).tolist(), train_y.tolist()))

    print('Number of nodes of oct', decision_tree.getNumNodes(oct_json))
    print('Number of nodes of cart', decision_tree.getNumNodes(cart_json))