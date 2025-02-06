import csv
import random
import math
import numpy as np

# Step 1: Data Loading
def load_data(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        data = [row for row in reader]
    return data[1:]  # Skip the header row

# Step 2: Data Preprocessing
def preprocess_data(data):
    # Convert Diagnosis column to numeric (N -> 0, O -> 1)
    for row in data:
        row[-1] = 0 if row[-1] == 'N' else 1

    # Convert all feature values to floats
    for row in data:
        for i in range(len(row) - 1):  # Exclude the target column
            row[i] = float(row[i])

    # Normalize features (Age, SittingHours)
    age_max, age_min = max(row[1] for row in data), min(row[1] for row in data)
    sitting_max, sitting_min = max(row[8] for row in data), min(row[8] for row in data)

    for row in data:
        row[1] = (row[1] - age_min) / (age_max - age_min)
        row[8] = (row[8] - sitting_min) / (sitting_max - sitting_min)

    return data


# Step 3: Train-Test Split
def custom_train_test_split(X, y, test_size=0.2):
    data = list(zip(X, y))
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    train, test = data[:split_idx], data[split_idx:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    # After splitting data
    print(f"X_train sample: {X_train[:2]}")
    print(f"Data types in X_train: {[type(feature) for feature in X_train[0]]}")
    return list(X_train), list(X_test), list(y_train), list(y_test)


# Step 4: Logistic Regression
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def logistic_regression_train(X_train, y_train, learning_rate=0.01, epochs=1000):
    weights = [random.random() for _ in range(len(X_train[0]))]
    for epoch in range(epochs):
        for i in range(len(X_train)):
            # Training Logistic Regression
            prediction = sigmoid(sum(X_train[i][j] * weights[j] for j in range(len(X_train[i]))))
            error = y_train[i] - prediction
            for j in range(len(weights)):
                weights[j] += learning_rate * error * X_train[i][j]
    return weights

def logistic_regression_predict(X_test, weights):
    return [1 if sigmoid(sum(X[j] * weights[j] for j in range(len(X)))) > 0.5 else 0 for X in X_test]

# Step 5: KNN
def euclidean_distance(x1, x2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = [(euclidean_distance(test_point, X_train[i]), y_train[i]) for i in range(len(X_train))]
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = [y for _, y in distances[:k]]
        predictions.append(max(set(nearest_neighbors), key=nearest_neighbors.count))
    return predictions

# K-fold Cross Validation
def k_fold_cross_validation(model_func, X, y, k=5, **kwargs):
    data = list(zip(X, y))
    random.shuffle(data)
    folds = [data[i::k] for i in range(k)]
    
    scores = []
    for i in range(k):
        test_fold = folds[i]
        train_folds = [fold for j, fold in enumerate(folds) if j != i]
        train_fold = [item for sublist in train_folds for item in sublist]

        X_train, y_train = zip(*train_fold)
        X_test, y_test = zip(*test_fold)

        if model_func == logistic_regression_train:
            weights = model_func(X_train, y_train, **kwargs)
            predictions = logistic_regression_predict(X_test, weights)
        elif model_func == knn_predict:
            predictions = model_func(list(X_train), list(y_train), list(X_test), **kwargs)
        
        accuracy = sum([p == y for p, y in zip(predictions, y_test)]) / len(y_test)
        scores.append(accuracy)

    return np.array(scores)

# Hyperparameter Optimization
def grid_search(model_func, X, y, param_grid, k=5):
    best_params = None
    best_score = -1

    for params in param_grid:
        scores = k_fold_cross_validation(model_func, X, y, k=k, **params)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    return best_params, best_score

# Main Code
data = load_data("fertility_data.txt")
processed_data = preprocess_data(data)
X = [row[:-1] for row in processed_data]
y = [row[-1] for row in processed_data]

# Perform Hyperparameter Optimization
logistic_param_grid = [{"learning_rate": lr, "epochs": ep} for lr in [0.001, 0.01, 0.1] for ep in [500, 1000, 1500]]
best_logistic_params, best_logistic_score = grid_search(logistic_regression_train, X, y, logistic_param_grid)
print(f"Best Logistic Regression Params: {best_logistic_params}, Best Score: {best_logistic_score:.2f}")

knn_param_grid = [{"k": k} for k in [1, 3, 5, 7, 9]]
best_knn_params, best_knn_score = grid_search(knn_predict, X, y, knn_param_grid)
print(f"Best KNN Params: {best_knn_params}, Best Score: {best_knn_score:.2f}")
