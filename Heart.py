import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
           'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
           'Oldpeak', 'Slope', 'MajorVessels', 'Thal', 'Target']
data = pd.read_csv(url, header=None, names=columns, na_values="?")

# Drop rows with missing values
data = data.dropna()

# Map categorical columns to numerical values
data['Sex'] = data['Sex'].astype(int)
data['ChestPainType'] = data['ChestPainType'].astype(int)
data['Thal'] = data['Thal'].astype(int)

# Binarize the target variable (0: No Heart Disease, 1: Heart Disease)
data['Target'] = (data['Target'] > 0).astype(int)

# Train-test split
X = data.drop('Target', axis=1)
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importances
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=True)
feature_importances.plot(kind='barh', color='skyblue')
plt.title("Feature Importance")
plt.show()

# Parameter Impact Analysis: Effect of max_depth on accuracy
max_depth_range = [3, 5, 10, 15, 20]
accuracies = []

for depth in max_depth_range:
    clf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

plt.plot(max_depth_range, accuracies, marker='o', color='r')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Effect of max_depth on Accuracy')
plt.show()

# Test the impact of min_samples_split
min_samples_split_range = [2, 5, 10, 20, 50]
accuracies_split = []

for split in min_samples_split_range:
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=split, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies_split.append(acc)

plt.plot(min_samples_split_range, accuracies_split, marker='o', color='b')
plt.xlabel('Min Samples Split')
plt.ylabel('Accuracy')
plt.title('Effect of min_samples_split on Accuracy')
plt.show()

# Test the impact of min_samples_leaf
min_samples_leaf_range = [1, 2, 5, 10, 30]
accuracies_leaf = []

for leaf in min_samples_leaf_range:
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=leaf, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies_leaf.append(acc)

plt.plot(min_samples_leaf_range, accuracies_leaf, marker='o', color='g')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Accuracy')
plt.title('Effect of min_samples_leaf on Accuracy')
plt.show()

# Test the impact of n_estimators
n_estimators_range = [10, 50, 100, 150, 200]
accuracies_estimators = []

for n in n_estimators_range:
    clf = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies_estimators.append(acc)

plt.plot(n_estimators_range, accuracies_estimators, marker='o', linestyle='-', color='purple')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Accuracy')
plt.title('Effect of n_estimators on Model Accuracy')
plt.show()

# Cross-Validation Scores
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Accuracy: {np.mean(cv_scores):.2f}')

plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='g')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores across Folds')
plt.show()
