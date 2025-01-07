import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Function
def sigmoid(z):
    """
    Compute the sigmoid of z.
    Sigmoid maps values to a range between 0 and 1.
    Formula: 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))

# Binary Cross-Entropy Loss
def cross_entropy_loss(y_true, y_pred):
    """
    Compute binary cross-entropy loss.
    Formula: -1/n * Σ(y*log(y_pred) + (1-y)*log(1-y_pred))
    """
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clipping predicted values
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Gradient Descent
def gradient_descent(X, y, weights, learning_rate, iterations):
    """
    Perform gradient descent to optimize weights.
    Update rule: w = w - learning_rate * gradient
    Gradient: (1/n) * X.T * (y_pred - y)
    """
    n = X.shape[0]  # Number of samples
    for i in range(iterations):
        # Predictions
        y_pred = sigmoid(np.dot(X, weights))
        # Compute gradient
        gradient = np.dot(X.T, (y_pred - y)) / n
        # Update weights
        weights -= learning_rate * gradient
        
        # Log loss for every 100 iterations
        if i % 100 == 0:
            loss = cross_entropy_loss(y, y_pred)
            print(f"Iteration {i}: Loss = {loss:.4f}")
    return weights

# Prediction Function
def predict(X, weights):
    """
    Predict class labels (0 or 1) based on a threshold of 0.5.
    """
    probabilities = sigmoid(np.dot(X, weights))
    return (probabilities >= 0.5).astype(int)

# Logistic Regression
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    """
    Fit logistic regression model using gradient descent.
    Returns optimized weights.
    """
    # Initialize weights to zeros
    weights = np.zeros(X.shape[1])
    # Optimize weights using gradient descent
    weights = gradient_descent(X, y, weights, learning_rate, iterations)
    return weights

# Evaluation Function
def evaluate(y_true, y_pred):
    """
    Compute accuracy of the model.
    Accuracy = (Number of correct predictions) / (Total predictions)
    """
    return np.mean(y_true == y_pred)

# Visualization of Decision Boundary
def plot_decision_boundary(X, y, weights):
    """
    Visualize the decision boundary for logistic regression.
    """
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x2 = -(weights[0] + weights[1] * x1) / weights[2]
    
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap='coolwarm', edgecolors='k', s=50)
    plt.plot(x1, x2, color="black", linestyle="--", label="Decision Boundary")
    plt.xlabel("Feature 1 (X1)")
    plt.ylabel("Feature 2 (X2)")
    plt.legend()
    plt.title("Logistic Regression Decision Boundary")
    plt.show()

# Data Preprocessing
def preprocess_data(X):
    """
    Normalize the dataset using standardization.
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)

# Main Program
# Dataset
X = np.array([
    [0.1, 1.1],
    [1.2, 0.9],
    [1.5, 1.6],
    [2.0, 1.8],
    [2.5, 2.1],
    [0.5, 1.5],
    [1.8, 2.3],
    [0.2, 0.7],
    [1.9, 1.4],
    [0.8, 0.6]
])

y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])  # Target values

# Add bias term (X0 = 1)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Normalize features
X[:, 1:] = preprocess_data(X[:, 1:])

# Plot the dataset
plt.scatter(X[:, 1], X[:, 2], c=y, cmap='coolwarm', edgecolors='k', s=50)
plt.xlabel("Feature 1 (X1)")
plt.ylabel("Feature 2 (X2)")
plt.title("Data Distribution")
plt.show()

# Train the logistic regression model
learning_rate = 0.1
iterations = 1000
weights = logistic_regression(X, y, learning_rate, iterations)

# Evaluate the model
y_pred = predict(X, weights)
accuracy = evaluate(y, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualize decision boundary
plot_decision_boundary(X, y, weights)
