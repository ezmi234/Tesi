import json
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import pandas as pd
from collections import deque

# # Read the JSON file
# with open('tree.json') as json_file:
#     tree_json = json.load(json_file)

# def build_sklearn_tree(node):
#     if node['type'] == 'leaf':
#         return _tree.Leaf(node['label'])
#     else:
#         left_child = build_sklearn_tree(node['left_child'])
#         right_child = build_sklearn_tree(node['right_child'])
#         return _tree.Node(node['feature'], left_child, right_child)

# # Build the sklearn-compatible tree
# sklearn_tree = build_sklearn_tree(tree_json)

# # Print the tree
# # feature_names = ["Feature1", "Feature2", "Feature3"]  # Replace with your feature names
# target_names = ["ClassA", "ClassB"]  # Replace with your target class names
# _tree.export_text(sklearn_tree, target_names=target_names)

feature_names = []

# def json_to_tree(node_data):
#     if node_data['type'] == 'feature':
#         feature_names.append(node_data['feature'])
#         node = DecisionTreeClassifier()
#         node.feature = node_data['feature']
#         node.left = json_to_tree(node_data['left_child'])
#         node.right = json_to_tree(node_data['right_child'])
#         return node
#     else:
#         return node_data['label']


def bfs_tree_predict(node_data, x): 
    queue = deque()
    queue.append(node_data)
    while len(queue) != 0:
        node = queue.popleft()
        if node['type'] == 'split':
            if int(x[node['split_feature']]) == 0:
                queue.append(node['left_child'])
            else:
                queue.append(node['right_child'])
        else:
            return node['label']

def test_tree_on_data(node_data, x, y):
    right = 0
    for i in range(len(x)):
        y_pred = bfs_tree_predict(node_data, x[i])
        if y_pred == int(y[i]):
            right += 1
        else:
            print("errore su")
            print(i)
        
    print(right / len(x))
    print(right)
    print(len(x)-right)
    
    
            


# Load the JSON data from a file
with open('tree.json', 'r') as json_file:
    tree_data = json.load(json_file)

# Convert JSON to tree
# tree = json_to_tree(tree_data)

x=[]
y=[]

with open('tiny.txt', 'r') as data_file:
    while True:
        line = data_file.readline()
        if not line:
            break
        x_temp = line.removesuffix('\n').split(' ')
        y_temp = x_temp.pop(len(x_temp) - 1)
        x.append(x_temp)
        y.append(y_temp)

# bfs_tree(tree_data)
test_tree_on_data(tree_data, x, y)


