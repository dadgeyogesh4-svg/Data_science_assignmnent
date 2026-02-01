# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:05:25 2023


"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Assuming your dataset is named 'forest_fire.csv', adjust the file name as needed
df = pd.read_csv('forestfires.csv')

# Display the first few rows of the dataset
print(df.head())

# Mapping 'Small' and 'Large' to 0 and 1 for binary classification
df['Size_Categorie'] = df['size_category'].map({'small': 0, 'large': 1})

# Define features (X) and target variable (y)
X = df.drop('Size_Categorie', axis=1)
y = df['size_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



numeric_features = X.select_dtypes(include=['float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM classifier
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Train the model
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_rep)
