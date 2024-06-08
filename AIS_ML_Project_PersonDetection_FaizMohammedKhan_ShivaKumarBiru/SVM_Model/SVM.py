import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import joblib

# Load NumPy arrays
X = np.load('C:/Users/faizm/DataSet/nps/features.npy', allow_pickle=True)
y = np.load('C:/Users/faizm/DataSet/nps/labels.npy', allow_pickle=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the SVM classifier
clf = make_pipeline(SimpleImputer(strategy='mean'), SVC(C=60.0))

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Save the trained classifier to a file
#joblib.dump(clf, 'C:/Users/faizm/DataSet/svm_classifier.pkl')

# Make predictions on the training and testing data
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Generate confusion matrices
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

# Plot confusion matrices
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Training Data')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Testing Data')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.tight_layout()
plt.show()
