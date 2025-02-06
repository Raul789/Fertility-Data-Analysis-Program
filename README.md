# Fertility Data Analysis Program

## Overview
This project is designed to analyze fertility-related data using machine learning techniques. It implements Logistic Regression and K-Nearest Neighbors (KNN) for predicting the fertility diagnosis (`N` → 0, `O` → 1) based on various features such as age, sitting hours, and more. The program also includes custom functions for data preprocessing, train-test splitting, k-fold cross-validation, and hyperparameter optimization.

The dataset used in this project is available at the UCI Machine Learning Repository:  
[https://archive.ics.uci.edu/dataset/244/fertility](https://archive.ics.uci.edu/dataset/244/fertility)

## Features
- **Data Loading**: Reads fertility data from a `.txt` file containing comma-separated values.
- **Data Preprocessing**:
  - Converts categorical labels (`N`, `O`) into numeric values.
  - Normalizes specific features (Age and Sitting Hours) to ensure consistent scaling.
- **Custom Train-Test Split**: Manually splits the data into training and testing sets.
- **Machine Learning Algorithms**:
  - **Logistic Regression**: Trains a logistic regression model using gradient descent to optimize weights.
  - **K-Nearest Neighbors (KNN)**: Classifies data points based on the majority vote of their nearest neighbors.
- **K-Fold Cross-Validation**: Evaluates both models’ accuracy across multiple data splits for more robust performance.
- **Hyperparameter Optimization**: Tests different combinations of learning rates, epochs, and k-values for improved performance.
- 
![Feature Importance Chart](img/FeatureImportance.png)

## License

Copyright (c) 2025 Turc Raul

All rights reserved. This project is for educational purposes only and may not be reused or redistributed without explicit permission.
