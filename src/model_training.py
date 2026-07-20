import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_base_models(X_train, y_train):
    """Trains 6 baseline regression models including XGBoost and runs 5-fold CV."""
    print("Training baseline models...")
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=6)
    }
    
    cv_results = {}
    for name, model in models.items():
        start_time = time.time()
        
        # Fit model
        model.fit(X_train, y_train)
        duration = time.time() - start_time
        
        # 5-fold Cross Validation (R2 score)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        mean_cv = np.mean(cv_scores)
        std_cv = np.std(cv_scores)
        cv_results[name] = {"cv_mean": mean_cv, "cv_std": std_cv}
        
        print(f"  Trained {name} in {duration:.2f} seconds | 5-Fold CV R2: {mean_cv:.4f} (+/- {std_cv:.4f})")
        
    return models, cv_results

def evaluate_models(models, X_test, y_test, cv_results=None):
    """Evaluates all trained models on regression metrics and integrates CV scores if available."""
    print("\n--- Evaluating Models ---")
    results = []
    
    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        row = {
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        }
        
        if cv_results and name in cv_results:
            row["CV R2 Mean"] = cv_results[name]["cv_mean"]
            row["CV R2 Std"] = cv_results[name]["cv_std"]
        else:
            row["CV R2 Mean"] = np.nan
            row["CV R2 Std"] = np.nan
            
        results.append(row)
        cv_str = f" | CV R2: {row['CV R2 Mean']:.4f}" if not np.isnan(row["CV R2 Mean"]) else ""
        print(f"  {name:20s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | R2: {r2:.4f}{cv_str}")
        
    return pd.DataFrame(results)

def tune_random_forest(X_train, y_train, sample_size=10000):
    """
    Performs hyperparameter tuning for Random Forest using both
    GridSearchCV and RandomizedSearchCV on a representative sample of training data.
    """
    print(f"\n--- Hyperparameter Tuning (Random Forest) on Sample of {sample_size} rows ---")
    
    # 1. Sample the training data to speed up hyperparameter search
    if len(X_train) > sample_size:
        indices = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_sample = X_train.iloc[indices] if isinstance(X_train, pd.DataFrame) else X_train[indices]
        y_sample = y_train.iloc[indices] if isinstance(y_train, pd.Series) else y_train[indices]
    else:
        X_sample, y_sample = X_train, y_train
        
    # Parameter grids
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestRegressor(random_state=42)
    
    # --- GridSearchCV ---
    print("Running GridSearchCV...")
    start_grid = time.time()
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_sample, y_sample)
    grid_duration = time.time() - start_grid
    print(f"GridSearchCV completed in {grid_duration:.2f} seconds.")
    print(f"Best Params (Grid): {grid_search.best_params_}")
    print(f"Best R2 Score (Grid): {grid_search.best_score_:.4f}")
    
    # --- RandomizedSearchCV ---
    print("\nRunning RandomizedSearchCV...")
    start_random = time.time()
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_sample, y_sample)
    random_duration = time.time() - start_random
    print(f"RandomizedSearchCV completed in {random_duration:.2f} seconds.")
    print(f"Best Params (Random): {random_search.best_params_}")
    print(f"Best R2 Score (Random): {random_search.best_score_:.4f}")
    
    # Determine the best hyperparameter set
    best_params = grid_search.best_params_ if grid_search.best_score_ >= random_search.best_score_ else random_search.best_params_
    
    tuning_info = {
        "grid_best_params": grid_search.best_params_,
        "grid_best_score": grid_search.best_score_,
        "grid_time": grid_duration,
        "random_best_params": random_search.best_params_,
        "random_best_score": random_search.best_score_,
        "random_time": random_duration,
        "selected_params": best_params
    }
    
    return best_params, tuning_info

def save_model(model, preprocessor, base_path):
    """Saves model and preprocessor to the specified directory."""
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    
    model_path = os.path.join(base_path, "models", "best_model.pkl")
    prep_path = os.path.join(base_path, "models", "preprocessor.pkl")
    scaler_path = os.path.join(base_path, "models", "scaler.pkl") # for backward compatibility
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, prep_path)
    joblib.dump(preprocessor, scaler_path)
    
    # Copy to root as well for backward compatibility
    root_model_path = os.path.join(base_path, "salary_predictor_model.pkl")
    joblib.dump(model, root_model_path)
    
    print(f"Best model saved to {model_path} and {root_model_path}")
    print(f"Preprocessor saved to {prep_path}")

