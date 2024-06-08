import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from convertToFFT import convert_to_fft

"""Program starts at main.py"""

def calculate_accuracy(sample_index, obj, predictions):
    true_label = np.argmax(obj.y_label[sample_index])
    predicted_label = np.argmax(predictions[sample_index])
    accuracy = 100 - (abs(true_label - predicted_label) / true_label * 100)
    return accuracy

clf = joblib.load('C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/randomForestClassifier_withThreshold.pkl')

test_datapath = [
                'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithoutThreshold/100/adc_100_constant_leaned.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithoutThreshold/100/adc_100_constant_normal.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithoutThreshold/100/adc_100_moving_leaned.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithoutThreshold/100/adc_100_moving_normal.xlsx',

                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithoutThreshold/110/adc_110_constant_leaned.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithoutThreshold/110/adc_110_constant_normal.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithoutThreshold/110/adc_110_moving_leaned.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithoutThreshold/110/adc_110_moving_normal.xlsx',

                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithThreshold/100/adc_100_constant_leaned.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithThreshold/100/adc_100_constant_normal.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithThreshold/100/adc_100_moving_leaned.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithThreshold/100/adc_100_Moving_normal.xlsx',

                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithThreshold/110/adc_110_constant_leaned.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithThreshold/110/adc_110_constant_normal.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithThreshold/110/adc_110_moving_leaned.xlsx',
                 'C:/Users/faizm/OneDrive/Desktop/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/TestData/WithThreshold/110/adc_110_moving_normal.xlsx',
                 ]


for file in test_datapath:
    data = pd.read_excel(file)

    data['label'] = 1

    no_headers = data.iloc[:, 16:]

    X = no_headers.drop(columns=['label']).to_numpy()
    labels = no_headers['label'].to_numpy()
    fft_df = convert_to_fft(X, labels)
    y_pred = clf.predict(fft_df)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    # Calculate accuracy
    accuracy = accuracy_score(labels, y_pred)
    print(f"Accuracy - {file.split('/')[-3]}/{file.split('/')[-2]}/{file.split('/')[-1]}: {accuracy*100} %")

    balanced_accuracy = balanced_accuracy_score(labels, y_pred)
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print("---------------------------------------------------")

# cm_test = confusion_matrix(labels, y_pred)

# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=True)
# plt.title('Confusion Matrix - Scenario Data')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')

# plt.tight_layout()
# plt.show()