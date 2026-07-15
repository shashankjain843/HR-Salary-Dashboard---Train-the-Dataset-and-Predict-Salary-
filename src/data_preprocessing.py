import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    """Loads dataset from CSV file path."""
    print("Loading data from file...")
    df = pd.read_csv(file_path)
    return df

def clean_data(df, target_col="Target_Salary"):
    """
    Cleans dataset by verifying and handling missing values, duplicates, 
    and physically impossible negative salaries.
    """
    print("\n--- Starting Data Cleaning ---")
    
    # 1. Check for Missing Values
    null_counts = df.isnull().sum()
    print("Missing Values Check:")
    for col, count in null_counts.items():
        print(f"  Column '{col}': {count} missing values")
        
    # 2. Check for Duplicate Records
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate records found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print("Dropped duplicate records.")
        
    # 3. Data Type Validation
    print("Checking Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  Column '{col}': {dtype}")

    # 4. Outliers / Invalid Value Check: Negative Salaries
    initial_rows = len(df)
    negative_salaries = df[df[target_col] < 0]
    num_negatives = len(negative_salaries)
    
    print(f"Negative salaries detected: {num_negatives} ({num_negatives/initial_rows * 100:.2f}%)")
    
    if num_negatives > 0:
        # Drop negative salaries as a physical salary cannot be negative
        df = df[df[target_col] >= 0].reset_index(drop=True)
        print(f"Cleaned dataset: Removed {num_negatives} rows with negative salaries.")
        print(f"Dataset size changed from {initial_rows} to {len(df)} rows.")
    
    # 5. Outlier Detection on Age & Experience
    # Validate that Age >= 18 and Experience >= 0
    invalid_age_exp = df[(df["Age"] < 18) | (df["Years_of_Experience"] < 0) | (df["Years_of_Experience"] > df["Age"] - 18)]
    print(f"Physically inconsistent Age/Experience rows: {len(invalid_age_exp)}")
    if len(invalid_age_exp) > 0:
        df = df[~df.index.isin(invalid_age_exp.index)].reset_index(drop=True)
        print(f"Removed {len(invalid_age_exp)} physically inconsistent rows.")
        
    print("--- Data Cleaning Completed Successfully ---\n")
    return df

def get_preprocessor(feature_cols):
    """
    Creates and returns a ColumnTransformer that scales numerical features.
    Strictly fit on training data and transform on test data to avoid leakage.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor
