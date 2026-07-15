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
    data type validation, outlier detection (IQR), and physical age-experience constraints.
    """
    print("\n--- Starting Data Cleaning ---")
    initial_rows = len(df)
    
    # 1. Check & Handle Missing Values
    null_counts = df.isnull().sum()
    print("Missing Values Check:")
    has_missing = False
    for col, count in null_counts.items():
        print(f"  Column '{col}': {count} missing values")
        if count > 0:
            has_missing = True
            
    if has_missing:
        # Drop rows with missing values as they represent a tiny portion or are essential for modeling
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
    print("Checking and Validating Data Types:")
    for col in df.columns:
        print(f"  Column '{col}': Current type is {df[col].dtype}")
    
    # Explicitly enforce numeric types
    try:
        df["Age"] = pd.to_numeric(df["Age"]).astype("int64")
        df["Years_of_Experience"] = pd.to_numeric(df["Years_of_Experience"]).astype("int64")
        df[target_col] = pd.to_numeric(df[target_col]).astype("float64")
        print("  [ACTION] Data types successfully validated and cast (Age: int64, Experience: int64, Salary: float64).")
    except Exception as e:
        print(f"  [WARNING] Data type casting failed: {str(e)}")
        # Drop rows that cannot be cast to numeric
        df = df.dropna(subset=["Age", "Years_of_Experience", target_col]).reset_index(drop=True)
        df["Age"] = df["Age"].astype("int64")
        df["Years_of_Experience"] = df["Years_of_Experience"].astype("int64")
        df[target_col] = df[target_col].astype("float64")
        print(f"  [ACTION] Kept and cast only numeric rows. Row count after: {len(df)}")

    # 4. Outlier Detection on Target Salary (using IQR)
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
    num_outliers = len(outliers)
    print(f"Target Salary Outliers (IQR Method):")
    print(f"  IQR: {iqr:.2f} | Lower Bound: {lower_bound:.2f} | Upper Bound: {upper_bound:.2f}")
    print(f"  Detected outliers: {num_outliers} ({num_outliers/len(df)*100:.2f}%)")
    print("  [DECISION] Outliers are kept as they represent valid high/low compensation cases and not errors.")

    # 5. Outliers / Invalid Value Check: Negative Salaries
    negative_salaries = df[df[target_col] < 0]
    num_negatives = len(negative_salaries)
    print(f"Negative salaries detected: {num_negatives} ({num_negatives/len(df) * 100:.2f}%)")
    if num_negatives > 0:
        df = df[df[target_col] >= 0].reset_index(drop=True)
        print(f"  [ACTION] Removed {num_negatives} rows with negative salaries.")
        print(f"  Row count after: {len(df)}")
    
    # 6. Physical Consistency Check (Age & Experience)
    # Rules: Age must be >= 18, Experience must be >= 0, and Experience must be <= Age - 18 (working age)
    invalid_age_exp = df[
        (df["Age"] < 18) | 
        (df["Years_of_Experience"] < 0) | 
        (df["Years_of_Experience"] > (df["Age"] - 18))
    ]
    num_invalid_rules = len(invalid_age_exp)
    print(f"Physically inconsistent Age/Experience rows: {num_invalid_rules}")
    if num_invalid_rules > 0:
        df = df[~df.index.isin(invalid_age_exp.index)].reset_index(drop=True)
        print(f"  [ACTION] Removed {num_invalid_rules} physically inconsistent rows.")
        print(f"  Row count after: {len(df)}")
        
    print(f"Final Cleaned Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Data Cleaning percentage reduction: {(initial_rows - len(df)) / initial_rows * 100:.2f}%")
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
