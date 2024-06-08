
# Project Title: Parameter Setting and Reliability Test of a Sensor System for Person Detection in a Car Wearing Winter Wear

### Authors:
- Shiva Kumar Biru 
- Faiz Mohammad Khan 

### Supervisors:
- Dr. Peter Nauth
- Dr. Andreas Pech

### Institution:
Frankfurt University of Applied Sciences

## Abstract
This project investigates the accurate detection of individuals within vehicles, especially under winter conditions where thick clothing poses challenges. The aim is to enhance safety and user experience in automotive systems by evaluating the performance of a sensor system in detecting individuals wearing winter clothing. Advanced machine learning algorithms such as Random Forest Classifier (RFC) and Support Vector Machines (SVM) were used. The study focuses on constructing confusion matrices for various scenarios, including empty car seats and individuals in winter attire.

## Keywords
- Red Pitaya
- Ultrasonic Sensor SRF02
- Fast Fourier Transform (FFT)
- Machine Learning
- Supervised Learning
- Convolutional Neural Networks (CNN)
- Random Forest Classifier (RFC)
- Confusion Matrix

## Table of Contents
- [Project Title: Parameter Setting and Reliability Test of a Sensor System for Person Detection in a Car Wearing Winter Wear](#project-title-parameter-setting-and-reliability-test-of-a-sensor-system-for-person-detection-in-a-car-wearing-winter-wear)
    - [Authors:](#authors)
    - [Supervisors:](#supervisors)
    - [Institution:](#institution)
  - [Abstract](#abstract)
  - [Keywords](#keywords)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Methodology](#methodology)
    - [Ultrasonic Sensor and Red Pitaya](#ultrasonic-sensor-and-red-pitaya)
    - [Analog to Digital Converter (ADC)](#analog-to-digital-converter-adc)
    - [Fast Fourier Transform (FFT)](#fast-fourier-transform-fft)
    - [Confusion Matrix](#confusion-matrix)
  - [Implementation](#implementation)
    - [Measurement Setup and Collection](#measurement-setup-and-collection)
    - [Data Collection](#data-collection)
  - [Results](#results)
  - [Conclusion](#conclusion)
  - [Contact](#contact)

## Introduction
The study aims to ensure the safety and reliability of sensor systems designed for detecting individuals in vehicles, especially during winter. It focuses on assessing the performance of the sensor system by examining how well it detects individuals wearing various types of winter clothing.

## Methodology
### Ultrasonic Sensor and Red Pitaya
The project employs the Ultrasonic Sensor SRF02 and the Red Pitaya STEM Lab board. The SRF02 sensor operates at a frequency of 40 kHz and is controlled by the Red Pitaya device, which also performs analog-to-digital conversion and facilitates wireless data transfer for further processing.

### Analog to Digital Converter (ADC)
The ADC is crucial for converting continuous analog signals into digital data, enabling the analysis of the captured signals, particularly for distance measurements.

### Fast Fourier Transform (FFT)
FFT is used to transform the time-domain signal into the frequency domain, allowing for more efficient data analysis and processing.

### Confusion Matrix
The confusion matrix is utilized to evaluate the performance of the classification models by comparing the predicted labels with the actual labels. Key metrics such as accuracy, recall, precision, and F1 score are derived from the confusion matrix.

## Implementation
### Measurement Setup and Collection
- **Measurement Environment:** Measurements were conducted using a Ford Fiesta parked in the basement garage of Frankfurt University of Applied Sciences.
- **Measurement Software:** Custom software developed at the university was used to process the data gathered from the Red Pitaya board.
- **Data Format:** The exported data includes text-based FFT and ADC data, with detailed headers providing useful information for analysis.

### Data Collection
Data was collected under various scenarios, including empty seats and seats occupied by individuals wearing winter clothing, at distances of 100 cm and 110 cm from the sensor.

[Data_files](https://github.com/shiva-kumar-biru/FRAUS_Machinelearning_AIS/tree/main/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru)

### Code

[RFC_Code](https://github.com/shiva-kumar-biru/FRAUS_Machinelearning_AIS/tree/main/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/RandomForest)

[SVM_Code](https://github.com/shiva-kumar-biru/FRAUS_Machinelearning_AIS/tree/main/AIS_ML_Project_PersonDetection_FaizMohammedKhan_ShivaKumarBiru/SVM_Model)

## Results
The results include detailed analyses of the sensor's performance across different scenarios, with emphasis on accuracy, reliability, and the impact of winter clothing on detection efficiency.

The Below Document has the project analysis and the result

[Document](https://github.com/shiva-kumar-biru/FRAUS_Machinelearning_AIS/blob/main/AIS_ML_Project_Report_FaizMohammedKhan_ShivaKumar_Biru.pdf)

## Conclusion
The study successfully demonstrated the capability of the sensor system to detect individuals in winter clothing, with the machine learning models providing robust performance in various scenarios.

## Contact
- **Shiva Kumar Biru:** shiva.biru@stud.fra-uas.de
- **Faiz Mohammad Khan:** faiz.khan@stud.fra-uas.de
