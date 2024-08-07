import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
url = 'https://www.kaggle.com/mlg-ulb/creditcardfraud/download'  # Replace with the path to your dataset
data = pd.read_csv('/Users/poojanpatel/Downloads/creditcard.csv')  # Replace 'creditcard.csv' with the path to your downloaded dataset

# Explore the dataset
print(data.head())
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Separate features and target variable
X = data.drop(columns=['Class'])
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
