import os
import json
import sys
import pandas as pd
import numpy as np

# Set standard output encoding to utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src.data_preprocessing import load_data, clean_data, get_preprocessor
from src.model_training import train_base_models, evaluate_models, tune_random_forest, save_model
from src.visualization import (
    plot_actual_vs_predicted, 
    plot_residual_plot, 
    plot_error_distribution, 
    plot_feature_importance,
    plot_learning_curve
)

def main():
    base_path = r"c:\Users\Shashank\OneDrive\ドキュメント\hr_sal_dashboard"
    raw_data_path = os.path.join(base_path, "data", "raw", "hr_salary_data.csv")
    processed_data_path = os.path.join(base_path, "data", "processed", "hr_salary_cleaned.csv")
    
    # 1. Load Data
    raw_df = load_data(raw_data_path)
    
    # 2. Clean Data
    cleaned_df = clean_data(raw_df)
    
    # Save Cleaned Data
    cleaned_df.to_csv(processed_data_path, index=False)
    print(f"Cleaned dataset saved to: {processed_data_path}")
    
    # 3. Features and Target split
    num_cols = ["Age", "Years_of_Experience", "Performance_Rating", "Certifications"]
    cat_cols = ["Job_Title", "Department", "Education_Level", "Location_Tier"]
    feature_cols = num_cols + cat_cols
    
    X = cleaned_df[feature_cols]
    y = cleaned_df["Target_Salary"]
    
    # 4. Train-Test Split (Preventing Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain size: {X_train.shape[0]} rows | Test size: {X_test.shape[0]} rows")
    print(f"Numerical Features: {num_cols} | Categorical Features: {cat_cols}")
    
    # 5. Preprocessing (Fit ColumnTransformer ONLY on training data)
    preprocessor = get_preprocessor(num_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # Extract feature names after One-Hot Encoding
    transformed_feature_names = preprocessor.get_feature_names_out()
    
    X_train_df = pd.DataFrame(X_train_trans, columns=transformed_feature_names)
    X_test_df = pd.DataFrame(X_test_trans, columns=transformed_feature_names)
    print("Features transformed successfully using ColumnTransformer (StandardScaler + OneHotEncoder).")
    
    # 6. Train baseline models
    models, cv_results = train_base_models(X_train_df, y_train)
    
    # 7. Evaluate baseline models
    metrics_df = evaluate_models(models, X_test_df, y_test, cv_results)
    metrics_df.to_csv(os.path.join(base_path, "reports", "model_comparison_metrics.csv"), index=False)
    
    # 8. Hyperparameter Tuning for Random Forest
    best_rf_params, tuning_info = tune_random_forest(X_train_df, y_train, sample_size=10000)
    
    print(f"\nTraining final Random Forest on full training data with tuned parameters: {best_rf_params}...")
    tuned_rf = RandomForestRegressor(**best_rf_params, random_state=42)
    tuned_rf.fit(X_train_df, y_train)
    
    print("Calculating cross-validation scores for Tuned Random Forest...")
    tuned_cv_scores = cross_val_score(tuned_rf, X_train_df, y_train, cv=5, scoring='r2', n_jobs=-1)
    cv_results["Tuned Random Forest"] = {
        "cv_mean": np.mean(tuned_cv_scores),
        "cv_std": np.std(tuned_cv_scores)
    }
    
    models["Tuned Random Forest"] = tuned_rf
    
    final_metrics_df = evaluate_models(models, X_test_df, y_test, cv_results)
    final_metrics_df.to_csv(os.path.join(base_path, "reports", "final_model_comparison_metrics.csv"), index=False)
    
    # 9. Diagnostics plot using best model
    best_row = final_metrics_df.sort_values(by="R2 Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = models[best_model_name]
    print(f"\nBest Model identified: {best_model_name} (R2 Score: {best_row['R2 Score']:.4f})")
    
    best_preds = best_model.predict(X_test_df)
    residuals = y_test - best_preds
    
    # 10. Generate evaluation plots
    print("\nGenerating evaluation plots to reports/ directory...")
    plot_actual_vs_predicted(
        y_test, best_preds, best_model_name, 
        save_path=os.path.join(base_path, "reports", "actual_vs_predicted.png")
    )
    plot_residual_plot(
        best_preds, residuals, best_model_name,
        save_path=os.path.join(base_path, "reports", "residuals_plot.png")
    )
    plot_error_distribution(
        residuals, best_model_name,
        save_path=os.path.join(base_path, "reports", "error_distribution.png")
    )
    plot_learning_curve(
        best_model, X_train_df, y_train, best_model_name,
        save_path=os.path.join(base_path, "reports", "learning_curve.png")
    )
    
    # Feature Importance (clean names)
    rf_tuned = models["Tuned Random Forest"]
    importances = rf_tuned.feature_importances_
    # Clean feature names for plot
    clean_feature_names = [f.replace('cat__', '').replace('num__', '') for f in transformed_feature_names]
    
    df_imp = plot_feature_importance(
        importances, clean_feature_names,
        save_path=os.path.join(base_path, "reports", "feature_importance.png")
    )
    df_imp.to_csv(os.path.join(base_path, "reports", "feature_importance.csv"), index=False)
    
    # 11. Save model and preprocessor
    save_model(best_model, preprocessor, base_path)
    
    metadata = {
        "best_model_name": best_model_name,
        "features": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "transformed_features": clean_feature_names,
        "metrics": final_metrics_df.to_dict(orient="records"),
        "tuning": tuning_info,
        "rf_importance": df_imp.to_dict(orient="records")
    }
    
    with open(os.path.join(base_path, "models", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\nPipeline execution complete! All artifacts and models saved.")


if __name__ == "__main__":
    main()
