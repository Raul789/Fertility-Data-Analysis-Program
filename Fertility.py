import csv
import random
import math
import numpy as np
from sklearn.model_selection import train_test_split


# Step 1: Data Loading (TXT)
def load_data(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')  # Specify comma as the delimiter
        data = [row for row in reader]
    return data[1:]  # Skip the header row

# Step 2: Data Preprocessing
def preprocess_data(data):
    # Convert Diagnosis column to numeric (N -> 0, O -> 1)
    for row in data:
        row[-1] = 0 if row[-1] == 'N' else 1

    # Normalize features (Age, SittingHours)
    age_max, age_min = max([float(row[1]) for row in data]), min([float(row[1]) for row in data])
    sitting_max, sitting_min = max([float(row[8]) for row in data]), min([float(row[8]) for row in data])

    # Convert the features to numeric values and normalize them
    for row in data:
        try:
            for i in range(len(row)):
                if isinstance(row[i], str):
                    row[i] = float(row[i])

            row[1] = (row[1] - age_min) / (age_max - age_min)
            row[8] = (row[8] - sitting_min) / (sitting_max - sitting_min)
        except ValueError:
            print(f"Error with row {row}, check for non-numeric values.")
    return data

# Step 3: Train-Test Split
def custom_train_test_split(X, y, test_size=0.2):
    data = list(zip(X, y))
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    train, test = data[:split_idx], data[split_idx:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return list(X_train), list(X_test), list(y_train), list(y_test)

# Step 4: Logistic Regression
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def logistic_regression_train(X_train, y_train, learning_rate=0.01, epochs=1000):
    weights = [random.random() for _ in range(len(X_train[0]))]
    for epoch in range(epochs):
        for i in range(len(X_train)):
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

# K-fold Cross Validation for Logistic Regression
def custom_k_fold(X, y, k=5):
    # Combine features and labels together
    data = list(zip(X, y))
    random.shuffle(data)  # Shuffle the data before splitting

    # Split data into k folds
    folds = [data[i::k] for i in range(k)]  # Create k folds

    scores = []
    for i in range(k):
        # Create the training and testing sets for the current fold
        test_fold = folds[i]
        train_folds = [fold for j, fold in enumerate(folds) if j != i]
        train_fold = [item for sublist in train_folds for item in sublist]
        
        X_train, y_train = zip(*train_fold)
        X_test, y_test = zip(*test_fold)

        # Train Logistic Regression and make predictions
        logistic_weights = logistic_regression_train(X_train, y_train)
        logistic_predictions = logistic_regression_predict(X_test, logistic_weights)

        # Calculate accuracy
        accuracy = sum([p == y for p, y in zip(logistic_predictions, y_test)]) / len(y_test)
        scores.append(accuracy)
    
    return np.array(scores)

# K-fold Cross Validation for KNN
def knn_k_fold(X, y, k=5):
    # Combine features and labels together
    data = list(zip(X, y))
    random.shuffle(data)  # Shuffle the data before splitting

    # Split data into k folds
    folds = [data[i::k] for i in range(k)]  # Create k folds

    scores = []
    for i in range(k):
        # Create the training and testing sets for the current fold
        test_fold = folds[i]
        train_folds = [fold for j, fold in enumerate(folds) if j != i]
        train_fold = [item for sublist in train_folds for item in sublist]

        X_train, y_train = zip(*train_fold)
        X_test, y_test = zip(*test_fold)

        # Train KNN and make predictions
        knn_predictions = knn_predict(list(X_train), list(y_train), list(X_test), k=3)
        
        # Calculate accuracy
        accuracy = sum([p == y for p, y in zip(knn_predictions, y_test)]) / len(y_test)
        scores.append(accuracy)

    return np.array(scores)

# Hyperparameter Optimization for Logistic Regression
def logistic_regression_grid_search(X, y, learning_rates, epochs_list, k=5):
    best_params = {}
    best_score = -1

    for learning_rate in learning_rates:
        for epochs in epochs_list:
            # Perform k-fold cross-validation
            scores = []
            for fold_idx in range(k):
                test_fold = folds[fold_idx]
                train_folds = [folds[i] for i in range(k) if i != fold_idx]
                train_fold = [item for sublist in train_folds for item in sublist]

                # Split into train/test for this fold
                X_train, y_train = zip(*train_fold)
                X_test, y_test = zip(*test_fold)

                # Train and evaluate logistic regression
                weights = logistic_regression_train(X_train, y_train, learning_rate, epochs)
                predictions = logistic_regression_predict(X_test, weights)
                accuracy = sum([pred == true for pred, true in zip(predictions, y_test)]) / len(y_test)
                scores.append(accuracy)

            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = {"learning_rate": learning_rate, "epochs": epochs}

    return best_params, best_score

# Hyperparameter Optimization for KNN
def knn_grid_search(X, y, k_values, k_folds=5):
    best_k = None
    best_score = -1

    for k in k_values:
        # Perform k-fold cross-validation
        scores = []
        for fold_idx in range(k_folds):
            test_fold = folds[fold_idx]
            train_folds = [folds[i] for i in range(k_folds) if i != fold_idx]
            train_fold = [item for sublist in train_folds for item in sublist]

            # Split into train/test for this fold
            X_train, y_train = zip(*train_fold)
            X_test, y_test = zip(*test_fold)

            # Train and evaluate KNN
            predictions = knn_predict(list(X_train), list(y_train), list(X_test), k)
            accuracy = sum([pred == true for pred, true in zip(predictions, y_test)]) / len(y_test)
            scores.append(accuracy)

        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_k = k

    return best_k, best_score



# Main Code
data = load_data("fertility_data.txt")
processed_data = preprocess_data(data)
X = [row[:-1] for row in processed_data]
y = [row[-1] for row in processed_data]
X_train, X_test, y_train, y_test = custom_train_test_split(X, y)

# Logistic Regression with K-fold Cross-Validation
logistic_scores = custom_k_fold(X, y, k=5)
print(f"Logistic Regression K-fold Accuracy: {logistic_scores.mean():.2f} ± {logistic_scores.std():.2f}")

# KNN with K-fold Cross-Validation
knn_scores = knn_k_fold(X, y, k=5)
print(f"KNN K-fold Accuracy: {knn_scores.mean():.2f} ± {knn_scores.std():.2f}")
