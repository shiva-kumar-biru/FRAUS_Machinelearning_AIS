import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from convertToFFT import convert_to_fft

"""Program starts at main.py"""

clf = joblib.load('C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/Dataset#2/numpy_arrays_classifier/rf_classifier.pkl')
test_datapath = 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithoutThreshold/110/adc_110_moving_normal.xlsx'

data = pd.read_excel(test_datapath)

data['label'] = 1

no_headers = data.iloc[:, 16:]

X = no_headers.drop(columns=['label']).to_numpy()
labels = no_headers['label'].to_numpy()
fft_df = convert_to_fft(X, labels)
y_pred = clf.predict(fft_df)

# Calculate accuracy
accuracy = accuracy_score(labels, y_pred)
print(f"Accuracy - {test_datapath.split('/')[-3]}/{test_datapath.split('/')[-2]}/{test_datapath.split('/')[-1]}: {accuracy*100} %")

# cm_test = confusion_matrix(labels, y_pred)

# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=True)
# plt.title('Confusion Matrix - Scenario Data')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')

# plt.tight_layout()
# plt.show()