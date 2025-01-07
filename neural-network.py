import numpy as np
import matplotlib.pyplot as plt

# Step 2: Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    """
    Initialize the weights and biases for the network.
    """
    np.random.seed(42)  # For reproducibility
    weights = {
        "W1": np.random.randn(hidden_size, input_size) * 0.01,  # Hidden layer weights
        "b1": np.zeros((hidden_size, 1)),                      # Hidden layer biases
        "W2": np.random.randn(output_size, hidden_size) * 0.01, # Output layer weights
        "b2": np.zeros((output_size, 1))                       # Output layer biases
    }
    return weights

# Step 3: Implement forward propagation
def forward_propagation(X, weights):
    """
    Compute the forward pass through the network.
    """
    W1, b1, W2, b2 = weights["W1"], weights["b1"], weights["W2"], weights["b2"]
    Z1 = np.dot(W1, X.T) + b1  # Hidden layer linear transform
    A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation
    Z2 = np.dot(W2, A1) + b2    # Output layer linear transform
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2.T, cache

# Step 4: Compute the loss
def compute_loss(y_true, y_pred):
    """
    Compute binary cross-entropy loss.
    """
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
    return loss

# Step 5: Implement backward propagation
def backward_propagation(X, y, weights, cache):
    """
    Compute gradients for backward propagation.
    """
    m = X.shape[0]
    A1, A2 = cache["A1"], cache["A2"]
    W2 = weights["W2"]

    # Compute gradients
    dZ2 = A2.T - y
    dW2 = np.dot(dZ2.T, A1.T) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True).T / m

    dZ1 = np.dot(dZ2, W2) * A1.T * (1 - A1.T)
    dW1 = np.dot(dZ1.T, X) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True).T / m

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients


# Step 6: Update weights
def update_parameters(weights, gradients, learning_rate):
    """
    Update the weights using gradient descent.
    """
    weights["W1"] -= learning_rate * gradients["dW1"]
    weights["b1"] -= learning_rate * gradients["db1"]
    weights["W2"] -= learning_rate * gradients["dW2"]
    weights["b2"] -= learning_rate * gradients["db2"]
    return weights

# Step 7: Training loop
def train_network(X, y, hidden_size, learning_rate, epochs):
    """
    Train the neural network.
    """
    # Ensure y is a column vector
    y = y.reshape(-1, 1)

    input_size = X.shape[1]
    output_size = 1
    weights = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Forward propagation
        y_pred, cache = forward_propagation(X, weights)

        # Compute loss
        loss = compute_loss(y, y_pred)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")

        # Backward propagation
        gradients = backward_propagation(X, y, weights, cache)

        # Update parameters
        weights = update_parameters(weights, gradients, learning_rate)

    return weights


# Step 8: Plot decision boundary
def plot_decision_boundary(X, y, weights):
    """
    Visualize the decision boundary of the trained network.
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict for the grid points
    y_pred, _ = forward_propagation(grid, weights)
    y_pred = (y_pred > 0.5).astype(int).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, alpha=0.8, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
    plt.title("Decision Boundary")
    plt.show()

# Dataset
X = np.array([
    [0.1, 0.6],
    [0.15, 0.71],
    [0.25, 0.8],
    [0.35, 0.45],
    [0.5, 0.5],
    [0.6, 0.2],
    [0.65, 0.3],
    [0.8, 0.35]
])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# Train the model
hidden_size = 3
learning_rate = 0.1
epochs = 5000
trained_weights = train_network(X, y, hidden_size, learning_rate, epochs)

# Visualize the results
plot_decision_boundary(X, y, trained_weights)
