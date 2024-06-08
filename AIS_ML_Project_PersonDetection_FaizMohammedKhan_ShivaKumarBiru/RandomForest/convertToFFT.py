import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


"""Program starts at main.py"""

# Load NumPy arrays
X = np.load('C:/Users/faizm/DataSet/nps/features_ds3.npy', allow_pickle=True)
y = np.load('C:/Users/faizm/DataSet/nps/labels2.npy', allow_pickle=True)

def convert_to_fft(X):

    adc_data_selected_columns = X.mean(axis=1)

    # Assuming `adc_data` is your pandas Series with ADC data
    adc_array = adc_data_selected_columns

    # Choose a window function - Hanning window in this case
    window = np.hanning(len(adc_array))

    # Apply the window function to your data
    windowed_adc_data = adc_array * window

    # Perform FFT on the windowed data
    fft_result = np.fft.fft(windowed_adc_data)

    # Frequency bins (assuming you know the sampling rate)
    sampling_rate = 1000  # Example: 1000 Hz, replace with your actual sampling rate
    n = len(adc_array)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    # Calculate the magnitude and phase of the FFT result
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)

    # Create a DataFrame
    fft_df = pd.DataFrame({
        'Frequency': freq,
        'FFT Magnitude': magnitude,
        'Phase': phase
    })

    fft_df.head()  # Display the first few rows of the DataFrame
    # numpy_array = fft_df.to_numpy()
    # # Save the array to a file
    # np.save('C:/Users/faizm/DataSet/nps/'+'fft_data_ds3.npy', numpy_array)
    # print('FFT saved')
    return fft_df.to_numpy()


# convert_to_fft(X, y)