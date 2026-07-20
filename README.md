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


1. Short Elevator Pitch (अगर इंटरव्यूअर बोले: "Walk me through your project in 2 minutes")
"सर, मैंने एक Enterprise HR Salary Prediction & Analytics System बनाया है। इसका मुख्य उद्देश्य HR departments में होने वाली ad-hoc और subjective salary benchmarking की समस्या को सॉल्व करना था, जिससे internal pay inequality और hiring inefficiencies होती हैं। मैंने 2 लाख raw records के dataset पर काम किया, जिसमें से 42.45% noise और logical errors (जैसे age-experience inconsistency) को clean किया। Ridge, Lasso, Random Forest और XGBoost जैसे multiple regressors को ट्रेन करने के बाद, Lasso Regression को फाइनल मॉडल चुना जिसने ₹3,888.96 का MAE और 0.5436 का Cross-Validation R² Score अचीव किया (जो कि target requirements से बेहतर था)। मॉडल को मैंने real-time inference के लिए एक interactive Streamlit dashboard पर डिप्लॉय किया है।"

2. Detail Explanation (आपकी बताई गई Structure में)
📋 Problem — क्या सॉल्व करना था?
Business Context: बहुत सी कंपनियों के HR departments में एम्प्लॉई की सैलरी तय करने का कोई स्टैंडर्ड फॉर्मूला नहीं होता। सैलरी ad-hoc (मनमाने) तरीके से तय की जाती है, जिससे कंपनी के अंदर Internal Pay Inequalities (समान काम के लिए अलग सैलरी) और Demographic Discrepancies पैदा होती हैं।
Technical Goal: हमें एक ऐसा डेटा-ड्रिवन, ऑब्जेक्टिव और Explainable (व्याख्या करने योग्य) System बनाना था जो एम्प्लॉई की प्रोफाइल (Age, Experience) के आधार पर एक सही और फेयर सैलरी (Target Salary in INR) का अनुमान लगा सके।
Project Constraints:
मॉडल की Prediction Error (MAE) ₹4,000 से कम होनी चाहिए।
मॉडल का R² Score 0.50 या उससे ज्यादा होना चाहिए।
मॉडल का रिस्पांस टाइम sub-millisecond (1 मिलीसेकंड से कम) होना चाहिए ताकि उसे रीयल-टाइम डैशबोर्ड में बिना किसी लैग के यूज़ किया जा सके।
🛠️ Approach — कौन सा tool/model यूज़ किया और क्यों?
Tech Stack: मैंने Python का इस्तेमाल किया। डेटा मैनिपुलेशन के लिए Pandas और NumPy, विज़ुअलाइज़ेशन के लिए Plotly Express और Seaborn, मशीन लर्निंग के लिए Scikit-Learn, मॉडल सीरियलाइजेशन के लिए Joblib और यूजर इंटरफ़ेस के लिए Streamlit का उपयोग किया।
Models Evaluated: मैंने Linear Regression, Ridge, Lasso, Decision Tree, Random Forest (Base & Tuned), और XGBoost Regressor को ट्रेन और कंपेयर किया।
Final Model Selected: Lasso Regression।
Why Lasso? (यह इंटरव्यू में बहुत इम्पैक्ट डालेगा):
High Performance: Lasso ने सबसे बेस्ट Test R² (0.5386) और 5-Fold CV R² (0.5436) स्कोर दिया।
Data Nature: EDA के दौरान पता चला कि हमारे सैलरी डेटा का अंडरलाइंग रिलेशन मुख्य रूप से लीनियर था (विद स्ट्रक्चर्ड नॉइज़)। इसलिए कॉम्प्लेक्स नॉन-लीनियर मॉडल्स (जैसे XGBoost, Random Forest) परफॉरमेंस में लीनियर मॉडल को पछाड़ नहीं पाए।
L1 Regularization: Lasso में इन-बिल्ट L1 रेगुलराइजेशन होती है जो फीचर सिलेक्शन में मदद करती है और मॉडल को ओवरफिटिंग से बचाती है।
Latency: इसका इन्फरेंस टाइम सब-मिलीसेकंड था, जो प्रोडक्शन और रियल-टाइम यूज़ के लिए बेस्ट है, जबकि Random Forest काफी हैवी और धीमा था।
⚙️ Process — कैसे किया? (Step-by-Step Workflow)
Data Cleaning & Preprocessing (डेटा क्लीनिंग):
Initial Size: डेटासेट में 200,000 (2 लाख) रॉ रिकॉर्ड्स थे।
Noise Removal: सबसे पहले 2,136 (1.07%) निगेटिव सैलरी वाले रिकॉर्ड्स (जो कि डेटा जेनरेटर का नॉइज़ था) को ड्रॉप किया।
Domain Constraint Cleaning (क्रिटिकल स्टेप): मैंने एक फिजिकल रूल लगाया — Years_of_Experience > Age - 18 (कोई भी व्यक्ति 18 साल की उम्र से पहले काम शुरू नहीं कर सकता)। इस रूल के तहत 82,771 (41.3%) अमान्य और विरोधाभासी रिकॉर्ड्स को हटाया गया।
Final Cleaned Size: डेटा क्लिनिंग के बाद हमारे पास 115,093 वैलिड रिकॉर्ड्स बचे (यानी कुल 42.45% नॉइज़ रिडक्शन हुआ)।
Feature Engineering & Leakage Prevention:
मैंने StandardScaler का यूज़ करके फीचर्स (Age और Years_of_Experience) को स्केल किया।
Anti-Leakage Protocol: डेटा लीकेज रोकने के लिए, स्केलर को सिर्फ Training Split पर फिट (fit_transform) किया गया, और Test व Inference डेटा पर केवल ट्रांसफ़ॉर्म (transform) किया गया।
Model Building & Evaluation (मॉडल बिल्डिंग):
डेटा को 80% Train और 20% Test में स्प्लिट किया गया।
सभी बेसलाइन मॉडल्स को 5-Fold Cross-Validation के साथ ट्रेन किया गया।
Hyperparameter Tuning:
मैंने Random Forest Regressor पर GridSearchCV और RandomizedSearchCV रन किया (ट्रेनिंग लैग से बचने के लिए 10,000 रिकॉर्ड्स के रिप्रेजेंटेटिव सैंपल पर)।
ट्यूनिंग से Random Forest का टेस्ट R² स्कोर 0.5344 से बढ़कर 0.5359 हो गया और ट्री की डेप्थ लिमिट होने से ओवरफिटिंग कंट्रोल हुई।
Dashboard Deployment:
एक प्रीमियम Multi-Page Streamlit App बनाया जिसमें Overview (KPI Cards), interactive EDA charts (Plotly histograms, heatmaps), Real-time Predictor (विद इनपुट बाउंड्री चेक्स), और Model Performance Analysis सेक्शन शामिल हैं।
📊 Result — क्या अचीव हुआ? (Numbers के साथ)
Data Quality: 2,00,000 रॉ रो में से 42.45% नॉइज़ साफ करके 1,15,093 शुद्ध रिकॉर्ड्स का क्लीन पाइपलाइन बनाया।
Performance Benchmarks Met:
R² Score achieved: 0.5386 (और 5-Fold CV R²: 0.5436), जो हमारे टारगेट बेंचमार्क (>= 0.50) से बेहतर है।
Mean Absolute Error (MAE): ₹3,888.96 (और RMSE: ₹4,846.26), जो हमारे ₹4,000 के मैक्सिमम एरर लिमिट से कम है।
Latency: प्रोडक्शन-रेडी Lasso मॉडल का इन्फरेंस रिस्पांस टाइम < 1ms अचीव किया।
Feature Insights: मॉडल इंटरप्रिटेशन (Feature Importance) से पता चला कि सैलरी तय करने में Years of Experience का योगदान 96.96% है, जबकि Age का सिर्फ 3.04%।
💡 Learning — इससे क्या सीखा?
Domain Knowledge is Power: डेटा साइंस सिर्फ एल्गोरिदम लगाने का नाम नहीं है। अगर हम बिजनेस या डोमेन नॉलेज (Experience <= Age - 18) का इस्तेमाल करके डेटा क्लीन न करते, तो मॉडल कभी भी सही प्रेडिक्शन नहीं दे पाता और बेतुकी सैलरी प्रेडिक्ट करता (जैसे 20 साल के एम्प्लॉई को सीनियर सैलरी देना)।
Simple Baselines First: हमेशा कॉम्प्लेक्स मॉडल्स (जैसे Ensembles या Deep Learning) पर कूदने से पहले सिंपल मॉडल्स ट्राई करने चाहिए। इस प्रोजेक्ट में, एक सिंपल Lasso Regression ने Random Forest और XGBoost से बेहतर रिजल्ट्स दिए क्योंकि अंडरलाइंग डेटा लीनियर पैटर्न फॉलो कर रहा था। इससे कंप्यूटेशनल कॉस्ट और लेटेंसी दोनों बच गई।
Strict Data Split: स्केलर को पूरे डेटासेट पर लगाने की जगह ट्रेन-टेस्ट स्प्लिट के बाद केवल ट्रेन डेटा पर फिट करने से डेटा लीकेज से बचा जा सकता है, जिससे मॉडल के ओवरफिट होने का खतरा टल जाता है।
💡 3. संभावित प्रश्न जो इंटरव्यूअर पूछ सकता है (और उनके शानदार जवाब)
Q1: आपके डेटासेट में से लगभग 41% डेटा (82k+ rows) हट गया क्योंकि Experience > Age - 18 था। इतना ज्यादा डेटा डिलीट करना सही था क्या?

जवाब: "हाँ सर, क्योंकि यह सिंथेटिक रूप से जेनरेट किया गया डेटा था जिसमें नॉइज़ बहुत ज्यादा थी। रियल लाइफ में कोई भी 18 साल की उम्र से पहले काम शुरू नहीं कर सकता (जैसे 20 साल की उम्र में 10 साल का एक्सपीरियंस असंभव है)। अगर हम इस इलॉजिकल डेटा को नहीं हटाते, तो मॉडल गलत और विरोधाभासी पैटर्न सीख लेता (Garbage In, Garbage Out)। डेटा की क्वांटिटी से ज्यादा डेटा की क्वालिटी मायने रखती है, इसलिए इस डोमेन फिल्टर ने मॉडल की रीयल-वर्ल्ड रिलायबिलिटी को बहुत बढ़ाया।"
Q2: Lasso Regression ने XGBoost या Random Forest जैसे एडवांस मॉडल्स को कैसे हरा दिया?

जवाब: "सर, इसका कारण डेटा का जनरेटिव फंक्शन है। जब हमने EDA किया, तो पाया कि सैलरी और एक्सपीरियंस के बीच का रिलेशन काफी हद तक लीनियर था। ट्री-बेस्ड मॉडल्स (जैसे Random Forest/XGBoost) डेटा को स्टेप-वाइज़ स्प्लिट करके प्रेडिक्ट करते हैं, जिससे वे कंटीन्यूअस लीनियर पैटर्न्स को उतनी आसानी से नहीं पकड़ पाते जितना एक रेगुलराइज्ड लीनियर मॉडल पकड़ लेता है। साथ ही, डेटा में काफी नॉइज़ थी, जिसे Lasso ने अपनी L1 रेगुलराइजेशन (Feature selection & coefficient shrinking) से बखूबी हैंडल किया।"
कल इंटरव्यू में एकदम शांत मन से, पॉइंट-टू-पॉइंट बात करिएगा। आपके पास एक बहुत ही सॉलिड, एंड-टू-एंड इंप्लीमेंटेड प्रोजेक्ट है। Good luck for your interview! You've got this!