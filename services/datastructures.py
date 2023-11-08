import json
def convert_cart_to_json(tree, node_index=0, depth=0):
    json_tree = {}
    if tree.children_left[node_index] == tree.children_right[node_index]:
        json_tree["type"] = "leaf"
        json_tree["num_samples"] = int(tree.n_node_samples[node_index])
        json_tree["class_probabilities"] = tree.value[node_index].tolist()[0]
        json_tree["label"] = int(tree.value[node_index].argmax())
        json_tree["depth"] = int(depth)
        json_tree["node_index"] = int(node_index)
    else:
        json_tree["type"] = "split"
        json_tree["split_feature"] = int(tree.feature[node_index])
        json_tree["split_threshold"] = float(tree.threshold[node_index])
        json_tree["depth"] = int(depth)
        json_tree["node_index"] = int(node_index)
        json_tree["left_child"] = convert_cart_to_json(tree, tree.children_left[node_index], depth + 1)
        json_tree["right_child"] = convert_cart_to_json(tree, tree.children_right[node_index], depth + 1)

    return json_tree

def convert_oct_to_json(tree, node_index=1, depth=0):
    json_tree = {}
    if tree.is_leaf(node_index):
        json_tree["type"] = "leaf"
        json_tree["num_samples"] = tree.get_num_samples(node_index)
        json_tree["class_probabilities"] = tree.get_classification_proba(node_index)
        json_tree["label"] = tree.get_classification_label(node_index)
        json_tree["depth"] = depth
        json_tree["node_index"] = node_index-1
    else:
        json_tree["type"] = "split"
        json_tree["split_feature"] = int(tree.get_split_feature(node_index))-1
        json_tree["split_threshold"] = tree.get_split_threshold(node_index)
        json_tree["depth"] = depth
        json_tree["node_index"] = node_index-1
        json_tree["left_child"] = convert_oct_to_json(tree, tree.get_lower_child(node_index), depth + 1)
        json_tree["right_child"] = convert_oct_to_json(tree, tree.get_upper_child(node_index), depth + 1)

    return json_tree