import pandas as pd
import numpy as np

# Read data from Excel files
passenger_data = pd.read_excel('C:/Users/faizm/OneDrive/Desktop/passenger_fft.xlsx')
empty_seat_data = pd.read_excel('C:/Users/faizm/OneDrive/Desktop/empty_fft.xlsx')

# Add label column
passenger_data['label'] = 1
empty_seat_data['label'] = 0

# Concatenate passenger and empty seat data
combined_data = pd.concat([passenger_data.iloc[:, 16:], empty_seat_data.iloc[:, 16:]], ignore_index=True)

# Shuffle the combined data
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert data to NumPy arrays
X = combined_data.drop(columns=['label']).to_numpy()
y = combined_data['label'].to_numpy()

# Confirm shapes
print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# Save NumPy arrays
#np.save('features.npy', X)
#np.save('labels.npy', y)

#np.save('C:/Users/faizm/DataSet/nps/' + 'features.npy', X)
#np.save('C:/Users/faizm/DataSet/nps/' + 'labels.npy', y)
