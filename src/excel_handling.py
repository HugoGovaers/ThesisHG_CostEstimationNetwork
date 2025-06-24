import os

import pandas as pd


def load_csv(file_path: str, sheet_name: str):
    """
    Loads data from a specified tab in an Excel file.

    Parameters:
    - file_path (str): Path to the Excel file.
    - sheet_name (str): Name of the sheet to load.

    Returns:
    - np.ndarray: A numpy array containing the data from the specified sheet.
    """
    try:
        # Load the specified sheet into a DataFrame
        df = pd.read_csv(file_path, sheet_name=sheet_name)
        # Convert DataFrame to numpy array
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
