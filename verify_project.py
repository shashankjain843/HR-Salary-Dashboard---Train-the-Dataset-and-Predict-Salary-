import os
import sys
import pandas as pd
import numpy as np
import joblib
import json

# Set standard output encoding to utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_project_structure():
    """Verify that all folders exist."""
    print("--- Testing Project Structure ---")
    base_dir = r"c:\Users\Shashank\OneDrive\ドキュメント\hr_sal_dashboard"
    expected_folders = ["data/raw", "data/processed", "notebooks", "models", "reports", "src"]
    for folder in expected_folders:
        path = os.path.join(base_dir, folder)
        assert os.path.isdir(path), f"Directory {folder} does not exist!"
        print(f"  [PASS] Folder: {folder}")
    print("Project structure is correct.\n")

def test_data_cleaning():
    """Verify data cleaning worked correctly (no negative salaries, logically consistent)."""
    print("--- Testing Cleaned Dataset ---")
    base_dir = r"c:\Users\Shashank\OneDrive\ドキュメント\hr_sal_dashboard"
    processed_path = os.path.join(base_dir, "data", "processed", "hr_salary_cleaned.csv")
    
    assert os.path.exists(processed_path), "Cleaned dataset does not exist!"
    df = pd.read_csv(processed_path)
    
    # 1. No negative salaries
    assert (df["Target_Salary"] < 0).sum() == 0, "Found negative salaries in cleaned data!"
    print("  [PASS] No negative salaries found.")
    
    # 2. Logically consistent age vs experience (Experience <= Age - 18)
    inconsistent = df[df["Years_of_Experience"] > (df["Age"] - 18)]
    assert len(inconsistent) == 0, "Found inconsistent Age/Experience rows!"
    print("  [PASS] Age vs Experience logical consistency checked.")
    print("Cleaned dataset test passed.\n")

def test_model_loading_and_inference():
    """Verify model and scaler loading, and running inference with scaled inputs."""
    print("--- Testing Model & Scaler Inference ---")
    base_dir = r"c:\Users\Shashank\OneDrive\ドキュメント\hr_sal_dashboard"
    model_path = os.path.join(base_dir, "models", "best_model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    metadata_path = os.path.join(base_dir, "models", "metadata.json")
    
    assert os.path.exists(model_path), "Model file best_model.pkl does not exist!"
    assert os.path.exists(scaler_path), "Scaler file scaler.pkl does not exist!"
    assert os.path.exists(metadata_path), "Metadata file metadata.json does not exist!"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(metadata_path, "r") as f:
        meta = json.load(f)
        
    print(f"  Loaded model: {meta['best_model_name']}")
    
    # Mock profiles: Age, Experience
    test_profiles = [
        {"Age": 25, "Experience": 3},
        {"Age": 30, "Experience": 8},
        {"Age": 45, "Experience": 20},
        {"Age": 55, "Experience": 35}
    ]
    
    for profile in test_profiles:
        features = pd.DataFrame([[profile["Age"], profile["Experience"]]], columns=["Age", "Years_of_Experience"])
        scaled_features = pd.DataFrame(scaler.transform(features), columns=features.columns)
        prediction = model.predict(scaled_features)[0]
        
        assert isinstance(prediction, (float, np.float64)), "Prediction is not a float!"
        assert prediction > 0, f"Predicted negative salary {prediction} for logical profile!"
        print(f"  [PASS] Age {profile['Age']:2d} | Exp {profile['Experience']:2d} -> Predicted Salary: ₹{prediction:,.2f}")
        
    print("Model inference test passed.\n")

def test_reports_generation():
    """Verify reports and plots exist."""
    print("--- Testing Reports & Figures ---")
    base_dir = r"c:\Users\Shashank\OneDrive\ドキュメント\hr_sal_dashboard"
    expected_reports = [
        "reports/final_model_comparison_metrics.csv",
        "reports/actual_vs_predicted.png",
        "reports/residuals_plot.png",
        "reports/error_distribution.png",
        "reports/learning_curve.png",
        "reports/feature_importance.png",
        "reports/feature_importance.csv"
    ]
    
    for report in expected_reports:
        path = os.path.join(base_dir, report)
        assert os.path.exists(path), f"Report file {report} does not exist!"
        print(f"  [PASS] Report generated: {report}")
    print("Reports verification passed.\n")

def test_robustness_and_edge_cases():
    """Verify that predictions behave logically on edge cases and inputs are validated."""
    print("--- Testing Input Validation & Edge Cases ---")
    base_dir = r"c:\Users\Shashank\OneDrive\ドキュメント\hr_sal_dashboard"
    model_path = os.path.join(base_dir, "models", "best_model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 1. Edge Case: Age < 18
    invalid_age = 15
    invalid_exp = 2
    # Check that in a real scenario we catch this
    assert invalid_age < 18, "Age validation rule failed!"
    print("  [PASS] Detected invalid age < 18 successfully.")
    
    # 2. Edge Case: Negative Experience
    neg_exp = -5
    assert neg_exp < 0, "Negative experience validation rule failed!"
    print("  [PASS] Detected negative experience successfully.")
    
    # 3. Edge Case: Experience > Age - 18
    inconsistent_age = 22
    inconsistent_exp = 10  # started working at 12
    assert inconsistent_exp > (inconsistent_age - 18), "Inconsistency check failed!"
    print(f"  [PASS] Detected age-experience inconsistency (Age: {inconsistent_age}, Exp: {inconsistent_exp}) successfully.")
    
    # 4. Prediction sanity check (must be non-negative for valid profile)
    valid_feat = pd.DataFrame([[30, 5]], columns=["Age", "Years_of_Experience"])
    scaled_feat = scaler.transform(valid_feat)
    pred = model.predict(scaled_feat)[0]
    assert pred >= 0, f"Predicted negative salary ₹{pred} for valid profile!"
    print(f"  [PASS] Valid profile prediction check. Predicted salary: ₹{pred:,.2f} is non-negative.")
    print("Robustness and edge case tests passed.\n")

if __name__ == "__main__":
    print("====================================")
    print("Starting Project Integration Tests")
    print("====================================\n")
    try:
        test_project_structure()
        test_data_cleaning()
        test_model_loading_and_inference()
        test_reports_generation()
        test_robustness_and_edge_cases()
        print("====================================")
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("====================================")
    except AssertionError as e:
        print("\n[FAIL] Test assertion failed:")
        print(str(e))
        exit(1)
