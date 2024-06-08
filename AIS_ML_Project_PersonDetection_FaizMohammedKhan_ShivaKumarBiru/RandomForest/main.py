from createData import create_data
from convertToFFT import convert_to_fft
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


"""Main Script for starting the program"""

passenger_paths = [
    "C:/Users/faizm/OneDrive/Desktop/DataSet#3/Person1/Jacket#1_100cms/adc_all_scenarios.xlsx",
    "C:/Users/faizm/OneDrive/Desktop/DataSet#3/Person1/Jacket#2_110cms/adc_all_scenarios.xlsx",

    "C:/Users/faizm/OneDrive/Desktop/DataSet#3/Person1/Jacket#3_110cms/adc_constant.xlsx",
    "C:/Users/faizm/OneDrive/Desktop/DataSet#3/Person1/Jacket#3_110cms/adc_moving.xlsx",

    "C:/Users/faizm/OneDrive/Desktop/DataSet#3/Person2/Jacket#1_100cms/adc_constant.xlsx",
    "C:/Users/faizm/OneDrive/Desktop/DataSet#3/Person2/Jacket#1_100cms/adc_moving.xlsx",
    "C:/Users/faizm/OneDrive/Desktop/DataSet#3/Person2/Jacket#2_110cms/adc_constant.xlsx",
    "C:/Users/faizm/OneDrive/Desktop/DataSet#3/Person2/Jacket#2_110cms/adc_moving.xlsx",
    "C:/Users/faizm/OneDrive/Desktop/DataSet#3/Person2/Jacket#3_110cms/adc_constant.xlsx",
    "C:/Users/faizm/OneDrive/Desktop/DataSet#3/Person2/Jacket#3_110cms/adc_moving.xlsx"
    # Add more file paths as needed
]

empty_seat_path = 'C:/Users/faizm/OneDrive/Desktop/DataSet#3/adc_empty.xlsx'

#Step 1 : Create Numpy arrays from xlsx data
(feature_path, label_path) = create_data(passenger_paths, empty_seat_path, 'feature_dataset3', 'labels_dataset3') # Pass in the data paths, empty seat path and name of the features and labels to be saved as

# Step 2
X = np.load(feature_path, allow_pickle=True)
fft_df = convert_to_fft(X) # Convert to fft and get the fft dataframe

# Step 3
# Assuming fft_df is your DataFrame containing FFT data
X = fft_df
y = np.load(label_path, allow_pickle=True)

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

# Create confusion matrix for training and test data
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

# Save the classifier
joblib.dump(rf_classifier, 'C:/Users/faizm/DataSet/rf_classifier_ds3_md.pkl')

# For predicting unseen data using the saved classifier, look into unseen_data.py