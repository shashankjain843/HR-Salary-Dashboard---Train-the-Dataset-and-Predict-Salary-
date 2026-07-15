import os
import json
import sys
import pandas as pd
import numpy as np

# Set standard output encoding to utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src.data_preprocessing import load_data, clean_data, get_preprocessor
from src.model_training import train_base_models, evaluate_models, tune_random_forest, save_model
from src.visualization import (
    plot_actual_vs_predicted, 
    plot_residual_plot, 
    plot_error_distribution, 
    plot_feature_importance
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
    feature_cols = ["Age", "Years_of_Experience"]
    X = cleaned_df[feature_cols]
    y = cleaned_df["Target_Salary"]
    
    # 4. Train-Test Split (Preventing Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain size: {X_train.shape[0]} rows | Test size: {X_test.shape[0]} rows")
    print(f"Features: {feature_cols} | Target: Target_Salary")
    
    # 5. Scaling features (Fit only on training data, transform test data)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
    print("Features scaled successfully using StandardScaler.")
    
    # 6. Train baseline models
    models = train_base_models(X_train_scaled, y_train)
    
    # 7. Evaluate baseline models
    metrics_df = evaluate_models(models, X_test_scaled, y_test)
    metrics_df.to_csv(os.path.join(base_path, "reports", "model_comparison_metrics.csv"), index=False)
    
    # 8. Hyperparameter Tuning for Random Forest
    # Run on a representative sample of training data (10,000 rows)
    best_rf_params, tuning_info = tune_random_forest(X_train_scaled, y_train, sample_size=10000)
    
    # Train final Random Forest with best parameters on the full training dataset
    print(f"\nTraining final Random Forest on full training data with tuned parameters: {best_rf_params}...")
    tuned_rf = RandomForestRegressor(**best_rf_params, random_state=42)
    tuned_rf.fit(X_train_scaled, y_train)
    
    # Update models dict with Tuned Random Forest
    models["Tuned Random Forest"] = tuned_rf
    
    # Re-evaluate all models (including tuned)
    final_metrics_df = evaluate_models(models, X_test_scaled, y_test)
    final_metrics_df.to_csv(os.path.join(base_path, "reports", "final_model_comparison_metrics.csv"), index=False)
    
    # 9. Get predictions for diagnostic plots using the best model
    # Check which model is best based on R2 Score
    best_row = final_metrics_df.sort_values(by="R2 Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = models[best_model_name]
    print(f"\nBest Model identified: {best_model_name} (R2 Score: {best_row['R2 Score']:.4f})")
    
    best_preds = best_model.predict(X_test_scaled)
    residuals = y_test - best_preds
    
    # 10. Generate and save diagnostic visualizations
    print("\nGenerating and saving evaluation plots to reports/ directory...")
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
    
    # Feature Importance for Random Forest (use tuned model)
    rf_tuned = models["Tuned Random Forest"]
    importances = rf_tuned.feature_importances_
    df_imp = plot_feature_importance(
        importances, feature_cols,
        save_path=os.path.join(base_path, "reports", "feature_importance.png")
    )
    df_imp.to_csv(os.path.join(base_path, "reports", "feature_importance.csv"), index=False)
    
    # 11. Save model and scaler
    save_model(best_model, scaler, base_path)
    
    # Save a JSON file with metadata for dashboard integration
    metadata = {
        "best_model_name": best_model_name,
        "features": feature_cols,
        "metrics": final_metrics_df.to_dict(orient="records"),
        "tuning": tuning_info,
        "rf_importance": df_imp.to_dict(orient="records")
    }
    
    with open(os.path.join(base_path, "models", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\nPipeline execution complete! All artifacts and models saved.")

if __name__ == "__main__":
    main()
