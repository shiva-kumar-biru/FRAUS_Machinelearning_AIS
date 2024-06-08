import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

"""Program starts at main.py"""

# Assuming fft_df is your DataFrame containing FFT data
X = np.load('C:/Users/faizm/DataSet/nps/fft_data_ds3.npy', allow_pickle=True)
y = np.load('C:/Users/faizm/DataSet/nps/labels_ds3.npy', allow_pickle=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=100)


rf_classifier.fit(X_train, y_train)

# Predict the labels
y_pred = rf_classifier.predict(X_test)
y_train_pred = rf_classifier.predict(X_train)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Testing Data)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Training Data)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

joblib.dump(rf_classifier, 'C:/Users/faizm/DataSet/rf_classifier_ds3_md.pkl')
