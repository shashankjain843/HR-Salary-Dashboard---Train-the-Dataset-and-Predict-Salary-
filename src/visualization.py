import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_actual_vs_predicted(y_test, preds, model_name, save_path=None):
    """Plots a scatter plot comparing actual and predicted values."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=preds, alpha=0.4, color="teal", edgecolor=None)
    
    # Plot diagonal line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit (y = x)")
    
    plt.title(f"{model_name}: Actual vs Predicted Salaries", fontsize=14, pad=15)
    plt.xlabel("Actual Salary (INR)", fontsize=12)
    plt.ylabel("Predicted Salary (INR)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()

def plot_residual_plot(preds, residuals, model_name, save_path=None):
    """Plots residuals vs predicted values to check model bias/variance."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=preds, y=residuals, alpha=0.4, color="purple", edgecolor=None)
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    
    plt.title(f"Residual Plot: {model_name}", fontsize=14, pad=15)
    plt.xlabel("Predicted Salary (INR)", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()

def plot_error_distribution(residuals, model_name, save_path=None):
    """Plots the distribution of residuals to check normality."""
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True, color="blue", stat="density", alpha=0.6)
    
    # Add a normal distribution comparison
    from scipy.stats import norm
    mu, std = norm.fit(residuals)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=rf"Normal Fit ($\mu={mu:.1f}$, $\sigma={std:.1f}$)")
    
    plt.title(f"Residual Error Distribution: {model_name}", fontsize=14, pad=15)
    plt.xlabel("Residuals (INR)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(importances, feature_names, save_path=None):
    """Plots a bar chart showing the relative feature importance."""
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Importance", y="Feature", data=df_imp, palette="viridis")
    
    # Annotate values
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=5)
        
    plt.title("Feature Importance Ranking", fontsize=14, pad=15)
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.xlim(0, 1.1)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()
        
    return df_imp
