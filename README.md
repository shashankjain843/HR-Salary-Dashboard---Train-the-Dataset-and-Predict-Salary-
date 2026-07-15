# Enterprise HR Salary Prediction & Analytics System

An industry-grade, end-to-end Data Science and Machine Learning project designed to clean payroll data, build robust regression models, perform hyperparameter optimization, and deploy an interactive analytics dashboard for real-time compensation benchmarking.

---

## 1. Project Overview
This project upgrades a basic regression project into a production-oriented, modular, and explainable Machine Learning system. The project features clean Python modules, a comprehensive Jupyter Notebook, multiple candidate models, hyperparameter tuning comparisons, and a premium interactive Streamlit dashboard.

---

## 2. Problem Statement
### Business Problem
determining employee salary benchmark structures is often done on an ad-hoc basis, leading to internal pay inequalities, demographic discrepancies, and hiring inefficiencies. HR departments require an objective, data-driven system to estimate fair compensation.

### Objectives
- **Build an End-to-End Pipeline**: Clean raw payroll datasets and enforce physical data rules (e.g. experience cannot exceed logical age limits).
- **Train Multiple Regressors**: Evaluate Linear Regression, Ridge, Lasso, Decision Trees, and Random Forests.
- **Hypertune Ensembles**: Compare GridSearchCV and RandomizedSearchCV for Random Forest Regressors.
- **Provide Dashboard Interface**: Deliver an interactive Streamlit UI containing KPI cards, Plotly charts, model metrics, and real-time inference.

### Expected Outcome
An explainable prediction system with less than 10% average error rate, deployed locally as a portfolio-ready web dashboard.

### Success Criteria
- **R² Score >= 0.50** (primary)
- **MAE < 4,000 INR** (secondary)

---

## 3. Dataset Description
- **Source**: Historical HR payroll database (synthetic representation).
- **Rows**: 200,000 records
- **Features**:
  - `Age` (integer): Employee age (range: 22 - 59).
  - `Years_of_Experience` (integer): Total years of experience (range: 0 - 39).
- **Target**: `Target_Salary` (float): The actual employee monthly salary in INR.

---

## 4. Technologies Used
- **Language**: Python 3.14
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Visualization**: Plotly Express, Matplotlib, Seaborn
- **Dashboard Deployment**: Streamlit
- **Serialization & Helper**: Joblib, Faker, Nbformat

---

## 5. Folder Structure
```
Salary_Prediction_Project/
│
├── data/
│   ├── raw/
│   │   └── hr_salary_data.csv          # Raw generated payroll data
│   └── processed/
│       └── hr_salary_cleaned.csv       # Preprocessed and filtered dataset
│
├── src/
│   ├── data_preprocessing.py           # Loading, cleaning, and scaling pipeline
│   ├── model_training.py               # Model fits, CV grid tuning, and saving
│   └── visualization.py                # Plotting diagnostics and importances
│
├── notebooks/
│   └── EDA_and_Model_Training.ipynb    # Comprehensive story-driven notebook
│
├── models/
│   ├── best_model.pkl                  # Serialized best regressor (Lasso)
│   ├── scaler.pkl                      # Fitted StandardScaler object
│   └── metadata.json                   # Aggregated metrics and parameter logs
│
├── reports/
│   ├── final_model_comparison_metrics.csv
│   ├── actual_vs_predicted.png
│   ├── residuals_plot.png
│   ├── error_distribution.png
│   └── feature_importance.png
│
├── app.py                              # Upgraded interactive Streamlit dashboard
├── requirements.txt                    # List of required package versions
├── verify_project.py                   # Automated validation and integration test
├── .gitignore                          # Standard git exclusions (models, venv)
└── README.md                           # Professional project documentation
```

---

## 6. Pipeline Workflow & Methodology

### Phase A: Dedicated Exploratory Data Analysis (EDA)
- **Dataset Dimensions**: Explicitly checked shapes (`df_raw.shape` shows 200,000 rows).
- **Statistical Summary**: Run `data.info()` and `data.describe()` to find range and columns. Minimum raw salary was negative (-13,727.57 INR), indicating synthetic generation noise.
- **Missing & Duplicate Analysis**: Checked for null values (none present) and duplicate records.
- **Correlations & Distributions**: Generated Correlation Matrix Heatmaps and univariate histograms. Years of Experience shows strong correlation (~0.73) with target salary.
- **Outlier Detection**: Built box plots showing salary ranges and identified target outliers (kept as valid high salary entries rather than errors).
- **Pair Plot Analysis**: Plotted a scatter matrix (pair plot) showing multi-feature relationships.

### Phase B: Rigorous Data Cleaning
- **Missing value handling**: Dropped if any were present (none in raw data).
- **Duplicate removal**: Deleted duplicated rows to prevent redundant records.
- **Data type validation**: Cast features to proper `int64` and target to `float64` for strict typing.
- **Invalid Values (Negative Salaries)**: Dropped 2,136 records (1.07%) with negative salaries.
- **Physical Inconsistencies**: Removed 82,771 records (41.3%) where `Years_of_Experience > Age - 18` (e.g. Experience exceeding legal adult working years).
- **Final Cleaned Dataset**: Verified dataset is completely clean. Final cleaned dataset dimensions: **115,093 rows x 3 columns** (representing 42.45% noise reduction).

### Phase C: Feature Engineering
- **Feature Scaling**: Normalized Age and Experience features using `StandardScaler`.
- **Feature Selection**: Correlation matrix confirmed no multicollinearity between Age and Experience.
- **Polynomial & Interaction Features**: Added markdown analysis explaining polynomial combinations and interactions (e.g. `Age * Experience`).
- **Strict Anti-Leakage Protocol**: Scaler fitted exclusively on training split, then applied independently to test and inference datasets.

### Phase D: Model Selection & Comparison
We train and evaluate 6 regressors using 5-fold Cross-Validation (CV) on the training set and standard evaluation on the test set:

| Model | MAE | RMSE | R² Score | 5-Fold CV R² Score |
|---|---|---|---|---|
| **Lasso Regression** | **₹3,888.96** | **₹4,846.26** | **0.5386** | **0.5436** |
| Ridge Regression | ₹3,888.96 | ₹4,846.26 | 0.5386 | 0.5436 |
| Linear Regression | ₹3,888.96 | ₹4,846.26 | 0.5386 | 0.5436 |
| XGBoost Regressor | ₹3,892.72 | ₹4,851.28 | 0.5376 | 0.5417 |
| Tuned Random Forest | ₹3,898.94 | ₹4,860.13 | 0.5359 | 0.5399 |
| Decision Tree | ₹3,904.38 | ₹4,867.85 | 0.5344 | 0.5382 |
| Random Forest (Base) | ₹3,904.57 | ₹4,867.98 | 0.5344 | 0.5381 |

*Note*: Lasso Regression was selected as the final production model due to highest R² and sub-millisecond inference latency.

### Phase E: Hyperparameter Tuning
Random Forest was tuned on a representative sample of 10,000 training records to prevent training lags:
- **GridSearchCV**: 108 fits completed in ~20s. Best params: `{'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 150}` (CV R² = 0.5023).
- **RandomizedSearchCV**: 30 fits completed in ~10s. Best params: `{'n_estimators': 150, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': 10}` (CV R² = 0.5023).
- **Before vs After Tuning (on Test Set)**:
  - Base RF: MAE = ₹3,904.57 | RMSE = ₹4,867.98 | R² = 0.5344
  - Tuned RF: MAE = ₹3,898.94 | RMSE = ₹4,860.13 | R² = 0.5359
  - *Result*: Tuning improved performance and successfully reduced tree overfitting by capping max depth.

### Phase F: Model Interpretation & Error Analysis
- **Feature Importance**: **Years of Experience** dominates predictions with **96.96%** of relative importance. **Age** holds only **3.04%** weight.
- **Error Analysis**: Predictions have higher absolute error for entry-level experience levels (0-5 Yrs) due to higher noise variance in synthetic generation.
- **Non-linear vs Linear Comparison**: Linear models (Lasso) perform slightly better than non-linear models (Decision Trees, Random Forest, XGBoost) because the underlying data generation function is strictly linear with structured noise.

---

## 7. Business Insights
1. **Experience is King**: Salary packages are overwhelmingly determined by actual years of professional experience rather than age.
2. **Quality Cleaning is Vital**: Removing physically inconsistent rows prevents the model from generating illogical predictions (such as a 20-year-old earning senior salary).
3. **Linear Baselines are Robust**: The linear models achieved an R² score of 0.5386, performing slightly better than ensembles due to the underlying linear function of the synthetic data.
4. **Diminishing Returns on Ensembles**: Random Forest is significantly heavier to train and tune, but yields no performance gain over Lasso in this linear scenario.
5. **Age-Experience Correlation Gap**: Age has a weak direct correlation with salary, meaning a mature career changer with 0 years of experience starts closer to entry-level salary levels.

---

## 8. Dashboard Features
- **Sidebar Navigation**: Select between Overview, EDA Explorer, Predictor, Model Performance, and About Project.
- **Dashboard KPIs**: Metric cards displaying Total cleaned records, Average salary, Salary range, and average experience/age.
- **Interactive Dataset Explorer**: Preview and compare the raw vs. cleaned datasets, check schema definitions, missing values, duplicates, and data quality logs.
- **Advanced EDA Tabs**: Interactive Plotly charts for histograms, heatmaps, outlier boxplots, and scatter matrices (pair plots).
- **Salary Predictor**: Text/Number inputs with built-in boundary checks and physical inconsistency alerts (warning if Experience > Age - 18).
- **Model Metrics & Tuning comparison**: Comparative performance table, R² & MAE plots, Grid vs Random tuning logs, and pre-rendered diagnostic charts (Residuals, Prediction Errors, and Learning Curves).
- **About Project**: A complete end-to-end documentation reference page in the dashboard.

---

## 9. Installation & How to Run

### Step 1: Clone or Open Workspace
Ensure you are in the project folder `c:\Users\Shashank\OneDrive\ドキュメント\hr_sal_dashboard`.

### Step 2: Initialize Virtual Environment
```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment
- **PowerShell (Windows)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Command Prompt (Windows)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run the End-to-End Pipeline
To clean data, train models, tune parameters, and generate report plots:
```bash
python run_pipeline.py
```

### Step 6: Run Integration Tests
To verify project integrity and check prediction logic:
```bash
python verify_project.py
```

### Step 7: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 10. Future Scope
- **Real Dataset Integration**: Train the model on authentic enterprise payroll records.
- **Explainable AI (XAI)**: Add SHAP and LIME summary plots to the dashboard to show exactly how much age vs experience contributed to a specific individual's prediction.
- **Database Integration**: Connect the backend to PostgreSQL or Snowflake for automated ingestion of real-time employee data.
- **API Deployment**: Deploy the model as a FastAPI microservice behind an authentication gateway.
- **Docker Support**: Containerize the app and database stack for seamless cloud deployment on GCP Cloud Run.
- **Additional Predictor Inputs**: Integrate Education Level, Performance Ratings, Department, and City Tier to improve predictive accuracy.

---

## 11. Author
*Senior Data Scientist & Machine Learning Engineer*
