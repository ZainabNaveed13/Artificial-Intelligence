import numpy as np
import matplotlib.pyplot as plt

# Perceptron Implementation
def perceptron(X, y, learning_rate, epochs):
    """
    Train a Perceptron model.
    Parameters:
        X: Feature matrix (numpy array)
        y: Labels (numpy array, 0 or 1)
        learning_rate: Step size for weight updates
        epochs: Number of training iterations
    Returns:
        weights: Learned weights
        bias: Learned bias
    """
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)  # Initialize weights to zero
    bias = 0  # Initialize bias to zero
    
    for epoch in range(epochs):
        for i in range(num_samples):
            # Calculate linear output
            linear_output = np.dot(X[i], weights) + bias
            # Apply step function
            y_pred = 1 if linear_output > 0 else 0
            # Update weights and bias if prediction is incorrect
            if y_pred != y[i]:
                weights += learning_rate * (y[i] - y_pred) * X[i]
                bias += learning_rate * (y[i] - y_pred)
    return weights, bias

# Visualization of Perceptron Decision Boundary
def plot_perceptron_boundary(X, y, weights, bias):
    """
    Visualize the decision boundary for the Perceptron.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z > 0, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title("Perceptron Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Generate dataset for Perceptron
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Linearly separable (AND gate)

# Train the Perceptron
weights, bias = perceptron(X, y, learning_rate=0.1, epochs=10)

# Visualize the decision boundary
plot_perceptron_boundary(X, y, weights, bias)

from sklearn.neural_network import MLPClassifier

# XOR Neural Network Implementation
def train_xor_nn(X, y):
    """
    Train a neural network to solve the XOR problem.
    """
    model = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=10000, random_state=0)
    model.fit(X, y)
    return model

# Visualization of XOR Decision Boundary
def visualize_xor_boundary(model, X, y):
    """
    Visualize the decision boundary for the XOR problem.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title("XOR Neural Network Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR gate

# Train the Neural Network
model = train_xor_nn(X, y)

# Visualize the XOR decision boundary
visualize_xor_boundary(model, X, y)
