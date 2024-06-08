import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

clf = joblib.load('C:/Users/faizm/DataSet/rf_classifier.pkl')

data = pd.read_excel('C:/Users/faizm/OneDrive/Desktop/DataSet#2/Faiz/Jacket#1_100cms/fft_all_scenarios.xlsx')

data['label'] = 1

no_headers = data.iloc[:, 16:]

X = no_headers.drop(columns=['label']).to_numpy()
labels = no_headers['label'].to_numpy()


y_pred = clf.predict(X)

cm_test = confusion_matrix(labels, y_pred)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Scenario Data')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.tight_layout()
plt.show()