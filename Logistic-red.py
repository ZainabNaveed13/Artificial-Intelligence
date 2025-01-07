import numpy as np

# Function to calculate the mean of a list or numpy array
def calculate_mean(values):
    return sum(values) / len(values)

# Function to calculate the slope (theta_1)
def calculate_slope(X, Y, mean_X, mean_Y):
    numerator = sum((X - mean_X) * (Y - mean_Y))
    denominator = sum((X - mean_X) ** 2)
    return numerator / denominator

# Function to calculate the intercept (theta_0)
def calculate_intercept(mean_X, mean_Y, slope):
    return mean_Y - slope * mean_X

# Function to predict Y values given X, theta_0, and theta_1
def predict(X, theta_0, theta_1):
    return theta_0 + theta_1 * X

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

# Function to perform Gradient Descent
def gradient_descent(X, Y, theta_0, theta_1, learning_rate, iterations):
    m = len(Y)
    for _ in range(iterations):
        # Predictions
        Y_pred = theta_0 + theta_1 * X
        
        # Calculate gradients
        gradient_theta_0 = -(2/m) * sum(Y - Y_pred)
        gradient_theta_1 = -(2/m) * sum((Y - Y_pred) * X)
        
        # Update weights
        theta_0 -= learning_rate * gradient_theta_0
        theta_1 -= learning_rate * gradient_theta_1
    
    return theta_0, theta_1

# Function to fit the linear regression model
def fit_linear_regression(X, Y, learning_rate=0.01, iterations=1000):
    # Calculate mean of X and Y
    mean_X = calculate_mean(X)
    mean_Y = calculate_mean(Y)
    
    # Calculate initial slope and intercept
    slope = calculate_slope(X, Y, mean_X, mean_Y)
    intercept = calculate_intercept(mean_X, mean_Y, slope)
    
    # Optimize using gradient descent
    theta_0, theta_1 = gradient_descent(X, Y, intercept, slope, learning_rate, iterations)
    return theta_0, theta_1

# Function to test the model
def test_model():
    # Dataset
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Y = np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000])
    
    # Hyperparameters
    learning_rate = 0.01
    iterations = 1000
    
    # Fit the model
    theta_0, theta_1 = fit_linear_regression(X, Y, learning_rate, iterations)
    
    # Predictions
    Y_pred = predict(X, theta_0, theta_1)
    
    # Evaluate the model
    mse = calculate_mse(Y, Y_pred)
    
    print("Optimal Intercept (theta_0):", theta_0)
    print("Optimal Slope (theta_1):", theta_1)
    print("Mean Squared Error (MSE):", mse)
    print("Predicted Values:", Y_pred)

# Run the test function
test_model()
