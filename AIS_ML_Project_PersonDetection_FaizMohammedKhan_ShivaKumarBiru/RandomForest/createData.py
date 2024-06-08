import pandas as pd
import numpy as np

"""Program starts at main.py"""

# Read data from Excel files

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

def create_data(passenger_paths, empty_seat_path, feature_name, label_name):
    # Initialize an empty list to store the DataFrames
    dfs = []

    # Loop through the file paths and append each DataFrame to the list
    for file in passenger_paths:
        print(file)
        df = pd.read_excel(file)
        df['label'] = 1
        dfs.append(df)

    # Concatenate all the DataFrames in the list
    combined_df = pd.concat(dfs, ignore_index=True)

    empty_seat_data = pd.read_excel(empty_seat_path)

    empty_seat_data['label'] = 0

    # Concatenate passenger and empty seat data
    combined_data = pd.concat([combined_df.iloc[:, 16:], empty_seat_data.iloc[:, 16:]], ignore_index=True)

    # Shuffle the combined data
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert data to NumPy arrays
    X = combined_data.drop(columns=['label']).to_numpy()
    y = combined_data['label'].to_numpy()

    # Confirm shapes
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)

    # Save NumPy arrays
    print('Saving the Numpy araays, please wait....')
    np.save('C:/Users/faizm/DataSet/nps/' + f'{feature_name}.npy', X)
    np.save('C:/Users/faizm/DataSet/nps/' + f'{label_name}.npy', y)
    print('Saved at : C:/Users/faizm/DataSet/nps/')
    return ('C:/Users/faizm/DataSet/nps/' + f'{feature_name}.npy', 'C:/Users/faizm/DataSet/nps/' + f'{label_name}.npy')
