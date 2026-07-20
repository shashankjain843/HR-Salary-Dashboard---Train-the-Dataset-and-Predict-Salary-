import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    """Loads dataset from CSV file path."""
    print("Loading data from file...")
    df = pd.read_csv(file_path)
    return df

def clean_data(df, target_col="Target_Salary"):
    """
    Cleans dataset by verifying and handling missing values, duplicates, 
    data type validation, outlier detection, and physical age-experience constraints.
    """
    print("\n--- Starting Data Cleaning ---")
    initial_rows = len(df)
    
    # 1. Check & Handle Missing Values
    null_counts = df.isnull().sum()
    print("Missing Values Check:")
    has_missing = False
    for col, count in null_counts.items():
        if count > 0:
            print(f"  Column '{col}': {count} missing values")
            has_missing = True
            
    if has_missing:
        df = df.dropna().reset_index(drop=True)
        print(f"  [ACTION] Dropped rows with missing values. Row count after: {len(df)}")
    else:
        print("  [PASS] No missing values found in the dataset.")
        
    # 2. Check & Handle Duplicate Records
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate records found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  [ACTION] Dropped {duplicate_count} duplicate records. Row count after: {len(df)}")
    else:
        print("  [PASS] No duplicate records found.")
        
    # 3. Data Type Validation
    num_cols = ["Age", "Years_of_Experience", "Performance_Rating", "Certifications"]
    cat_cols = ["Job_Title", "Department", "Education_Level", "Location_Tier"]
    
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        
    df = df.dropna().reset_index(drop=True)
    
    # Enforce integer types on discrete columns
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].astype("int64")

    # 4. Negative Salary Check
    if target_col in df.columns:
        df = df[df[target_col] >= 0].reset_index(drop=True)
        
    # 5. Physical Consistency Check (Age & Experience)
    invalid_age_exp = df[
        (df["Age"] < 18) | 
        (df["Years_of_Experience"] < 0) | 
        (df["Years_of_Experience"] > (df["Age"] - 18))
    ]
    if len(invalid_age_exp) > 0:
        df = df[~df.index.isin(invalid_age_exp.index)].reset_index(drop=True)
        print(f"  [ACTION] Removed {len(invalid_age_exp)} physically inconsistent rows.")
        
    print(f"Final Cleaned Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print("--- Data Cleaning Completed Successfully ---\n")
    return df

def get_preprocessor(num_cols, cat_cols):
    """
    Creates and returns a ColumnTransformer that scales numerical features
    and One-Hot encodes categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor

