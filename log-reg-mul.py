import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Data Preparation
# ----------------------------

# Dataset (Replace this with your uploaded data if needed)
data = np.array([
    [7.781169, -8.501326, 2],
    [-5.196946, -8.801356, 0],
    [-3.351567, 9.732630, 1],
    [6.520819, -10.602400, 2],
    [-5.679012, -6.046427, 0],
    [-2.721297, 9.079178, 1],
    [-4.252581, -7.059616, 0],
    [-4.022774, 8.908089, 1],
    [8.291942, -8.426824, 2],
    [-6.230867, -7.066018, 0]
])

# Extract features and target
X = data[:, :2]  # Features: X1 and X2
y = data[:, 2].astype(int)  # Target: Class labels (0, 1, 2)

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept term (bias) by adding a column of ones
X_scaled_intercept = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

# ----------------------------
# 2. Helper Functions
# ----------------------------

def softmax(z):
    """
    Compute the softmax of z.

    Parameters:
    z (numpy.ndarray): Input array (logits).

    Returns:
    numpy.ndarray: Softmax probabilities for each class.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss_multiclass(y_true, y_pred):
    """
    Compute cross-entropy loss for multiclass classification.

    Parameters:
    y_true (numpy.ndarray): One-hot encoded actual labels.
    y_pred (numpy.ndarray): Predicted probabilities for each class.

    Returns:
    float: Cross-entropy loss.
    """
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def predict_multiclass(X, weights):
    """
    Predict class probabilities using softmax function.

    Parameters:
    X (numpy.ndarray): Feature matrix.
    weights (numpy.ndarray): Weight matrix (one column per class).

    Returns:
    numpy.ndarray: Predicted probabilities for each class.
    """
    z = np.dot(X, weights)
    return softmax(z)

def gradient_descent_multiclass(X, y, weights, learning_rate, iterations):
    """
    Perform gradient descent to optimize weights for multiclass classification.

    Parameters:
    X (numpy.ndarray): Feature matrix with intercept.
    y (numpy.ndarray): One-hot encoded target vector.
    weights (numpy.ndarray): Initial weight matrix.
    learning_rate (float): Learning rate for updates.
    iterations (int): Number of iterations.

    Returns:
    numpy.ndarray: Optimized weights.
    """
    m = len(y)
    for i in range(iterations):
        # Predictions
        y_pred = predict_multiclass(X, weights)
        
        # Calculate gradients
        gradient = (1/m) * np.dot(X.T, (y_pred - y))
        
        # Update weights
        weights -= learning_rate * gradient
        
    return weights

def one_hot_encode(y, num_classes):
    """
    One-hot encode the target vector.

    Parameters:
    y (numpy.ndarray): Target vector with class labels.
    num_classes (int): Number of classes.

    Returns:
    numpy.ndarray: One-hot encoded matrix.
    """
    return np.eye(num_classes)[y]

# ----------------------------
# 3. Visualization Functions
# ----------------------------

def plot_data_multiclass(X, y):
    """
    Plot the data points on a 2D graph with different colors for classes.

    Parameters:
    X (numpy.ndarray): Original feature matrix (not scaled).
    y (numpy.ndarray): Target vector.
    """
    plt.figure(figsize=(8,6))
    for cls in np.unique(y):
        plt.scatter(X[y==cls, 0], X[y==cls, 1], label=f'Class {cls}')
    plt.xlabel('Feature 1 (X1)')
    plt.ylabel('Feature 2 (X2)')
    plt.title('Original Data (Not Scaled)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_decision_boundary_multiclass(X, y, weights, scaler):
    """
    Plot the decision boundaries for multiclass classification.

    Parameters:
    X (numpy.ndarray): Original feature matrix (not scaled).
    y (numpy.ndarray): Target vector.
    weights (numpy.ndarray): Trained weights.
    scaler (StandardScaler): Fitted scaler for standardizing data.
    """
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    grid_scaled_intercept = np.hstack([np.ones((grid_scaled.shape[0], 1)), grid_scaled])
    
    Z = predict_multiclass(grid_scaled_intercept, weights)
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plot_data_multiclass(X, y)

# ----------------------------
# 4. Main Execution
# ----------------------------

def main():
    num_classes = 3
    y_one_hot = one_hot_encode(y, num_classes)
    
    learning_rate = 0.1
    iterations = 1000
    
    weights = np.zeros((X_scaled_intercept.shape[1], num_classes))
    weights = gradient_descent_multiclass(X_scaled_intercept, y_one_hot, weights, learning_rate, iterations)
    
    y_pred_prob = predict_multiclass(X_scaled_intercept, weights)
    loss = cross_entropy_loss_multiclass(y_one_hot, y_pred_prob)
    print(f"Cross-Entropy Loss: {loss:.4f}")
    
    plot_decision_boundary_multiclass(X, y, weights, scaler)

if __name__ == "__main__":
    main()