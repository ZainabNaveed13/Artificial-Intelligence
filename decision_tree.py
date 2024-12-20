import math
from collections import Counter

def calculate_entropy(data, target_col):
    values = [row[target_col] for row in data]
    total_count = len(values)
    value_counts = Counter(values)
    entropy = -sum((count / total_count) * math.log2(count / total_count) for count in value_counts.values())
    return entropy

def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    values = [row[attribute] for row in data]
    total_count = len(values)
    value_counts = Counter(values)
    
    weighted_entropy = sum(
        (value_counts[value] / total_count) * calculate_entropy(
            [row for row in data if row[attribute] == value],
            target_col
        )
        for value in value_counts
    )
    return total_entropy - weighted_entropy

def build_tree(data, attributes, target_col):
    values = [row[target_col] for row in data]
    if len(set(values)) == 1:
        return values[0]
    if not attributes:
        return Counter(values).most_common(1)[0][0]
    
    gains = {attr: calculate_information_gain(data, attr, target_col) for attr in attributes}
    best_attribute = max(gains, key=gains.get)
    
    tree = {best_attribute: {}}
    
    attribute_values = set(row[best_attribute] for row in data)
    for value in attribute_values:
        subset = [row for row in data if row[best_attribute] == value]
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]
        subtree = build_tree(subset, remaining_attributes, target_col)
        tree[best_attribute][value] = subtree
    
    return tree

def predict(tree, data_point):
    if not isinstance(tree, dict): 
        return tree
    attribute = next(iter(tree))
    subtree = tree[attribute].get(data_point[attribute])
    if not subtree: 
        return None
    return predict(subtree, data_point)

dataset = [
    {'Weather': 'Sunny', 'Temperature': 'Hot', 'Play?': 'No'},
    {'Weather': 'Overcast', 'Temperature': 'Hot', 'Play?': 'Yes'},
    {'Weather': 'Rainy', 'Temperature': 'Mild', 'Play?': 'Yes'},
]
attributes = ['Weather', 'Temperature']
target_col = 'Play?'

decision_tree = build_tree(dataset, attributes, target_col)

test_data = {'Weather': 'Sunny', 'Temperature': 'Mild'}
prediction = predict(decision_tree, test_data)

print("Decision Tree:", decision_tree)
print("Prediction for test data:", prediction)
