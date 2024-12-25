import math
from collections import Counter

def calculate_entropy(data, target_col):
    target_values = [row[target_col] for row in data]
    value_counts = Counter(target_values)
    total = len(data)
    entropy = -sum((count / total) * math.log2(count / total) for count in value_counts.values())
    return entropy

def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    total = len(data)
    attribute_values = set(row[attribute] for row in data)
    weighted_entropy = 0

    for value in attribute_values:
        subset = [row for row in data if row[attribute] == value]
        subset_entropy = calculate_entropy(subset, target_col)
        weighted_entropy += (len(subset) / total) * subset_entropy

    info_gain = total_entropy - weighted_entropy
    return info_gain

def build_tree(data, attributes, target_col, depth=0, max_depth=3):
    target_values = [row[target_col] for row in data]
    if len(set(target_values)) == 1:  
        return target_values[0]
    if not attributes or depth == max_depth:  
        return Counter(target_values).most_common(1)[0][0]

    gains = {attr: calculate_information_gain(data, attr, target_col) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}
    for value in set(row[best_attr] for row in data):
        subset = [row for row in data if row[best_attr] == value]
        subtree = build_tree(subset, [a for a in attributes if a != best_attr], target_col, depth + 1, max_depth)
        tree[best_attr][value] = subtree

    return tree

def predict(tree, data_point):
    if not isinstance(tree, dict):  
        return tree
    attr = next(iter(tree))
    value = data_point.get(attr)
    subtree = tree[attr].get(value)
    if subtree is None:
        return None  
    return predict(subtree, data_point)

def build_random_forest(data, attributes, target_col, n_trees=2):
    import random
    forest = []
    for _ in range(n_trees):
        subset = random.choices(data, k=len(data)) 
        tree = build_tree(subset, attributes, target_col)
        forest.append(tree)
    return forest

def predict_random_forest(forest, data_point):
    predictions = [predict(tree, data_point) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]

data = [
    {"Weather": "Sunny", "Temperature": "Hot", "Play?": "No"},
    {"Weather": "Overcast", "Temperature": "Hot", "Play?": "Yes"},
    {"Weather": "Rainy", "Temperature": "Mild", "Play?": "Yes"},
    {"Weather": "Sunny", "Temperature": "Mild", "Play?": "No"},
    {"Weather": "Overcast", "Temperature": "Mild", "Play?": "Yes"},
    {"Weather": "Rainy", "Temperature": "Hot", "Play?": "No"},
]

attributes = ["Weather", "Temperature"]
target_col = "Play?"
forest = build_random_forest(data, attributes, target_col)

new_data_point = {"Weather": "Sunny", "Temperature": "Hot"}
prediction = predict_random_forest(forest, new_data_point)
print("Prediction for new data point:", prediction)
