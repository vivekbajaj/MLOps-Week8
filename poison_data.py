# poison_data.py
# This script poisons the Iris dataset by replacing a specified percentage of
# the feature data with random (but valid) values. This simulates subtle, hard-to-detect data quality issues.

import pandas as pd
import numpy as np
import argparse

def poison_data(level):
    """
    Loads the iris.csv data, poisons a specified percentage of rows with random
    numbers (within the valid feature ranges), and overwrites the original file.

    Args:
        level (int): The percentage of data rows to poison (e.g., 5, 10, 50).
    """
    if not 0 <= level <= 100:
        raise ValueError("Poisoning level must be between 0 and 100.")

    file_path = 'data/iris.csv'
    print(f"Loading original data from {file_path}...")
    df = pd.read_csv(file_path)

    if level == 0:
        print("Poisoning level is 0. No changes will be made.")
        return

    # Determine the number of rows to poison
    n_rows = len(df)
    n_to_poison = int((level / 100) * n_rows)
    print(f"Poisoning {n_to_poison} out of {n_rows} rows ({level}%).")

    # Get random indices to poison
    poison_indices = np.random.choice(df.index, size=n_to_poison, replace=False)

    # Define the feature columns to be poisoned
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Define valid ranges for each feature, matching your validation
    feature_ranges = {
        'sepal_length': (4.0, 8.0),
        'sepal_width':  (2.0, 5.0),
        'petal_length': (1.0, 7.0),
        'petal_width':  (0.1, 3.0)
    }

    # Generate in-range random data for the selected rows
    random_features = np.column_stack([
        np.random.uniform(*feature_ranges['sepal_length'], n_to_poison),
        np.random.uniform(*feature_ranges['sepal_width'], n_to_poison),
        np.random.uniform(*feature_ranges['petal_length'], n_to_poison),
        np.random.uniform(*feature_ranges['petal_width'], n_to_poison),
    ])

    # Replace the original data at the selected indices
    df.loc[poison_indices, feature_columns] = random_features

    # Save the poisoned data, overwriting the original file
    df.to_csv(file_path, index=False)
    print(f"Successfully poisoned the data and saved the new version to {file_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Poison the Iris dataset with in-range, subtly corrupt random values.")
    parser.add_argument(
        '--level',
        type=int,
        required=True,
        help="The percentage of data to poison (e.g., 5, 10, 50)."
    )
    args = parser.parse_args()
    poison_data(args.level)
