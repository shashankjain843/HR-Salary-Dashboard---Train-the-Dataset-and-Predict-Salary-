import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="HR Salary Prediction & Analytics System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #4B5563 !important;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
    }
    
    /* Header Styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E293B;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    
    /* Predictions Display */
    .prediction-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 2px solid #3B82F6;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 15px -3px rgb(59 130 246 / 0.1);
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E40AF;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    .sidebar-text {
        font-size: 0.85rem;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data and cache it
@st.cache_data
def load_processed_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

# Load metadata JSON
@st.cache_data
def load_metadata(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

# Load model and scaler
@st.cache_resource
def load_model_artifacts(model_path, scaler_path):
    model = None
    scaler = None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    return model, scaler

# Define file paths
BASE_DIR = r"c:\Users\Shashank\OneDrive\ドキュメント\hr_sal_dashboard"
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "hr_salary_cleaned.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "hr_salary_data.csv")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load assets
df_cleaned = load_processed_data(CLEANED_DATA_PATH)
metadata = load_metadata(METADATA_PATH)
model, scaler = load_model_artifacts(MODEL_PATH, SCALER_PATH)

# If best model isn't saved, load the fallback root model
if model is None:
    fallback_model_path = os.path.join(BASE_DIR, "salary_predictor_model.pkl")
    if os.path.exists(fallback_model_path):
        model = joblib.load(fallback_model_path)

# Load raw dataset
df_raw = load_raw_data(RAW_DATA_PATH) if 'load_raw_data' in globals() else None
if df_raw is None:
    @st.cache_data
    def load_raw_data(file_path):
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None
    df_raw = load_raw_data(RAW_DATA_PATH)

# Sidebar Navigation & Information
st.sidebar.markdown("### 💼 HR Salary Prediction & Analytics")
st.sidebar.markdown("---")

# Navigation menu
menu_option = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home & Overview",
        "📊 Data Explorer & EDA",
        "🔮 Salary Predictor",
        "📈 Model Performance & Tuning",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

# Model details in sidebar
if metadata:
    st.sidebar.markdown(f"**Best Model:** `{metadata.get('best_model_name', 'Unknown')}`")
    st.sidebar.markdown(f"**Target Variable:** `Target_Salary` (INR)")
    st.sidebar.markdown("**Features Used:**")
    for feat in metadata.get("features", []):
        st.sidebar.markdown(f"- `{feat}`")
else:
    st.sidebar.markdown("**Model State:** Standard model loaded (unscaled)")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='sidebar-text'>This tool uses historical data to benchmark salary packages based on objective metrics.</div>", 
    unsafe_allow_html=True
)

# Header Section
st.markdown('<div class="main-header">HR Salary Analytics & Prediction Platform</div>', unsafe_allow_html=True)

# ----------------- HOME & OVERVIEW PAGE -----------------
if menu_option == "🏠 Home & Overview":
    st.markdown('<div class="section-header">Project Overview & Objectives</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(
            "This enterprise-level platform utilizes historical payroll datasets to build robust, "
            "explainable predictive models for HR salaries. By automating compensation benchmarking, "
            "organizations can ensure fairness, reduce compensation discrepancies, and streamline hiring negotiations."
        )
        st.markdown("""
        ### Core Business Objectives:
        - **Fair Compensation Benchmarking**: Standardize payroll structures based on objective parameters (Age, Experience).
        - **Data Leakage & Inconsistency Protection**: Filter out illogical data entries where work experience exceeds physical age constraints.
        - **Model Reliability**: Compare baseline regression models (Linear, Ridge, Lasso) with decision-based ensembles (Decision Tree, Random Forest, XGBoost) and select the optimal predictor.
        - **Hyperparameter Optimization**: Perform systematic grid and randomized parameter tuning to maximize R² score.
        """)
        
    with col2:
        st.info(
            "💡 **Success Criteria**:\n\n"
            "- **R² Score >= 0.50**: Explain at least 50% of salary variance using features.\n"
            "- **MAE < 4,000 INR**: Keep the average prediction error low relative to salaries."
        )

    # KPI Metrics Section
    st.markdown('<div class="section-header">Dashboard KPIs (Cleaned Payroll Data)</div>', unsafe_allow_html=True)
    
    if df_cleaned is not None:
        num_records = len(df_cleaned)
        avg_salary = df_cleaned["Target_Salary"].mean()
        max_salary = df_cleaned["Target_Salary"].max()
        min_salary = df_cleaned["Target_Salary"].min()
        avg_experience = df_cleaned["Years_of_Experience"].mean()
        avg_age = df_cleaned["Age"].mean()
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        
        with kpi_col1:
            st.metric(label="Total Cleaned Records", value=f"{num_records:,}")
        with kpi_col2:
            st.metric(label="Average Salary", value=f"₹{avg_salary:,.2f}")
        with kpi_col3:
            st.metric(label="Salary Range (Min - Max)", value=f"₹{min_salary:,.0f} - ₹{max_salary:,.0f}")
        with kpi_col4:
            st.metric(label="Avg Experience", value=f"{avg_experience:.1f} Yrs")
        with kpi_col5:
            st.metric(label="Average Age", value=f"{avg_age:.1f} Yrs")
    else:
        st.warning("Processed dataset not found! Please run the pipeline script `run_pipeline.py` first.")

    # Dataset Preview Tab
    st.markdown('<div class="section-header">Dataset Preview & Schema Documentation</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📄 First 10 Sample Rows", "📊 Column Definitions & Types"])
    
    with tab1:
        if df_cleaned is not None:
            data_choice = st.radio("Select dataset version to preview:", ["Cleaned Dataset", "Raw Dataset"], horizontal=True)
            if data_choice == "Cleaned Dataset":
                st.dataframe(df_cleaned.head(10), use_container_width=True)
                st.caption(f"Showing first 10 rows. Cleaned dataset dimensions: **{df_cleaned.shape[0]} rows x {df_cleaned.shape[1]} columns**")
            elif df_raw is not None:
                st.dataframe(df_raw.head(10), use_container_width=True)
                st.caption(f"Showing first 10 rows. Raw dataset dimensions: **{df_raw.shape[0]} rows x {df_raw.shape[1]} columns**")
            else:
                st.info("Raw dataset file is not available in raw directory.")
        else:
            st.error("No dataset available to preview.")
            
    with tab2:
        schema_df = pd.DataFrame({
            "Column Name": ["Age", "Years_of_Experience", "Target_Salary"],
            "Data Type": ["int64 (Integer)", "int64 (Integer)", "float64 (Decimal)"],
            "Role": ["Numerical Feature", "Numerical Feature", "Target Variable"],
            "Description": [
                "The age of the HR professional (range: 22 to 59 years).",
                "Total working experience in years (range: 0 to 39 years).",
                "The target monthly/annual salary in INR (physically cleaned to be positive)."
            ]
        })
        st.table(schema_df)

# ----------------- DATA EXPLORER & EDA PAGE -----------------
elif menu_option == "📊 Data Explorer & EDA":
    st.markdown('<div class="section-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    
    if df_cleaned is not None:
        # Distribution tab
        dist_tab, corr_tab, rel_tab, outlier_tab, pair_tab, info_tab = st.tabs([
            "📈 Feature Distributions", 
            "🔥 Correlation Heatmap", 
            "🔗 Relationship Analysis", 
            "🚫 Outlier Analysis",
            "🧬 Pair Plot (Scatter Matrix)",
            "📋 Data Quality & Info"
        ])
        
        with dist_tab:
            st.write("### Numerical Feature Distributions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_age = px.histogram(df_cleaned, x="Age", nbins=20, title="Age Distribution", color_discrete_sequence=["#3B82F6"])
                st.plotly_chart(fig_age, use_container_width=True)
                st.caption("Distribution of employee ages. Age ranges uniformly between 22 and 59.")
                
            with col2:
                fig_exp = px.histogram(df_cleaned, x="Years_of_Experience", nbins=20, title="Experience Distribution", color_discrete_sequence=["#10B981"])
                st.plotly_chart(fig_exp, use_container_width=True)
                st.caption("Distribution of experience. Uniform distribution up to 39 years.")
                
            with col3:
                fig_sal = px.histogram(df_cleaned, x="Target_Salary", nbins=30, title="Salary Distribution", color_discrete_sequence=["#EC4899"])
                st.plotly_chart(fig_sal, use_container_width=True)
                st.caption("Salary distribution. Approximately normal, centered around 16.8k INR.")
                
        with corr_tab:
            st.write("### Feature Correlations")
            corr = df_cleaned.corr()
            
            fig_heat = px.imshow(
                corr, 
                text_auto=".3f", 
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap Matrix",
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown(
                "#### Insights:\n"
                "- **Years of Experience** has a very strong positive correlation with **Target Salary** (~0.73).\n"
                "- **Age** has a moderate positive correlation with salary (~0.23).\n"
                "- Age and Experience show low correlation due to random sampling, but are bound by physical constraints."
            )
            
        with rel_tab:
            st.write("### Bivariate Relationships")
            sample_size = st.slider("Select sample size for plotting scatter plot", 1000, 20000, 5000, step=1000)
            df_sample = df_cleaned.sample(sample_size, random_state=42)
            
            fig_scatter = px.scatter(
                df_sample, 
                x="Years_of_Experience", 
                y="Target_Salary", 
                color="Age",
                color_continuous_scale="viridis",
                title=f"Years of Experience vs Target Salary (Sample of {sample_size:,} records)",
                opacity=0.6
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Salary increases consistently as experience increases. The gradient color shows that higher ages also slightly shift predictions upwards.")
            
        with outlier_tab:
            st.write("### Outlier Detection")
            col1, col2 = st.columns(2)
            with col1:
                fig_box1 = px.box(df_cleaned, y=["Age", "Years_of_Experience"], title="Boxplot: Age & Experience")
                st.plotly_chart(fig_box1, use_container_width=True)
            with col2:
                fig_box2 = px.box(df_cleaned, y="Target_Salary", title="Boxplot: Target Salary")
                st.plotly_chart(fig_box2, use_container_width=True)
            st.markdown(
                "No outliers are present in the feature columns (uniform distribution bounds). "
                "The target salary variable has an expected normal spread with outliers detected beyond "
                "Q3 + 1.5 * IQR (which are valid, high salary values and are preserved in modeling)."
            )

        with pair_tab:
            st.write("### Pair Plot (Scatter Plot Matrix)")
            df_pair_sample = df_cleaned.sample(min(2000, len(df_cleaned)), random_state=42)
            fig_pair = px.scatter_matrix(
                df_pair_sample,
                dimensions=["Age", "Years_of_Experience", "Target_Salary"],
                color="Age",
                color_continuous_scale="viridis",
                title="Pair Plot Scatter Matrix (2,000 Sample Rows)"
            )
            fig_pair.update_traces(diagonal_visible=False)
            st.plotly_chart(fig_pair, use_container_width=True)
            st.caption("Diagonal plots show univariate distributions. Non-diagonal plots show scatter relationship comparisons.")
            
        with info_tab:
            st.write("### Data Quality Overview")
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown("**Cleaned Dataset Information:**")
                # Construct data info dataframe
                info_data = {
                    "Column Name": df_cleaned.columns,
                    "Non-Null Count": [df_cleaned[col].notnull().sum() for col in df_cleaned.columns],
                    "Data Type": [str(df_cleaned[col].dtype) for col in df_cleaned.columns],
                    "Unique Values": [df_cleaned[col].nunique() for col in df_cleaned.columns]
                }
                st.dataframe(pd.DataFrame(info_data), use_container_width=True)
                
            with col_info2:
                st.markdown("**Missing & Duplicate Summary:**")
                nulls_cleaned = df_cleaned.isnull().sum()
                dup_cleaned = df_cleaned.duplicated().sum()
                
                qa_df = pd.DataFrame({
                    "Quality Metric": ["Missing Values (Age)", "Missing Values (Experience)", "Missing Values (Salary)", "Duplicate Records"],
                    "Count": [nulls_cleaned["Age"], nulls_cleaned["Years_of_Experience"], nulls_cleaned["Target_Salary"], dup_cleaned],
                    "Status": ["✅ Pass" if val == 0 else "❌ Fail" for val in [nulls_cleaned["Age"], nulls_cleaned["Years_of_Experience"], nulls_cleaned["Target_Salary"], dup_cleaned]]
                })
                st.table(qa_df)
                
            st.write("### Data Cleaning Summary & Validation")
            if df_raw is not None:
                raw_len = len(df_raw)
                cleaned_len = len(df_cleaned)
                removed_rows = raw_len - cleaned_len
                
                st.markdown(f"""
                - **Raw Dataset Size**: {raw_len:,} rows
                - **Cleaned Dataset Size**: {cleaned_len:,} rows
                - **Total Rows Dropped**: {removed_rows:,} ({removed_rows / raw_len * 100:.2f}%)
                
                **Rules Applied in Cleaning:**
                1. **Negative Salary Filter**: Dropped salaries < 0 INR (synthetic generation artifacts).
                2. **Physical Age-Experience Constraints**: Dropped records where `Years_of_Experience > Age - 18` (working age limit constraint).
                """)
    else:
        st.warning("Cleaned data not found. Please run pipeline first.")

# ----------------- SALARY PREDICTOR PAGE -----------------
elif menu_option == "🔮 Salary Predictor":
    st.markdown('<div class="section-header">Expected Salary Predictor</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Trained model could not be loaded. Please ensure `salary_predictor_model.pkl` or `models/best_model.pkl` is trained.")
    else:
        st.write("Enter the parameters below to predict the expected compensation benchmarking range for the HR professional:")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Input Professional Profile")
            age = st.number_input("Employee Age (Years):", min_value=18, max_value=65, value=30, step=1)
            years_experience = st.number_input("Years of Experience:", min_value=0, max_value=40, value=5, step=1)
            
            # Input validations and warning checks
            is_valid = True
            if age < 18 or age > 65:
                st.error("❌ **Invalid Input**: Age must be between 18 and 65 years.")
                is_valid = False
            if years_experience < 0 or years_experience > 40:
                st.error("❌ **Invalid Input**: Years of Experience must be between 0 and 40 years.")
                is_valid = False
                
            if is_valid and (years_experience > (age - 18)):
                st.warning(
                    f"⚠️ **Physical Inconsistency Alert**: An employee aged **{age}** cannot have "
                    f"**{years_experience}** years of experience (started working at age {age - years_experience}, which is younger than 18). "
                    f"The prediction might be extrapolated outside logical boundaries."
                )

        with col2:
            st.markdown("### Prediction Results")
            
            # Predict button
            predict_btn = st.button("Predict Salary", use_container_width=True, type="primary", disabled=not is_valid)
            
            if predict_btn and is_valid:
                # Prepare inputs
                try:
                    if scaler is not None:
                        # Scaling is required for the best model
                        features = pd.DataFrame([[age, years_experience]], columns=["Age", "Years_of_Experience"])
                        scaled_features = scaler.transform(features)
                        prediction = model.predict(scaled_features)
                    else:
                        # Model expects unscaled inputs (fallback)
                        prediction = model.predict([[age, years_experience]])
                    
                    predicted_salary = max(0.0, prediction[0]) # Clip negative values if any
                    
                    st.markdown(
                        f'<div class="prediction-box">'
                        f'<p style="margin:0; font-size:1rem; color:#4B5563; font-weight:600;">BENCHMARKED SALARY RECOMMENDATION</p>'
                        f'<p class="prediction-value">₹{predicted_salary:,.2f}</p>'
                        f'<p style="margin:0; font-size:0.85rem; color:#059669; font-weight:600;">'
                        f'✅ Benchmarked successfully using {metadata.get("best_model_name", "Linear Model") if metadata else "Trained Model"}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Summary card
                    with st.expander("🔍 Prediction Details & Equation Summary", expanded=True):
                        st.write(f"- **Input Profile:** Age = {age} years, Experience = {years_experience} years.")
                        if metadata:
                            st.write(f"- **Underlying Model:** {metadata['best_model_name']}")
                            st.write("- **Primary Driver:** Years of Experience (represents approx. 97% of predictor weight).")
                        st.write("- **Confidence Boundary:** ± ₹4,846.26 INR (based on RMSE score of the model).")
                except Exception as ex:
                    st.error(f"Prediction failed with error: {str(ex)}")
            else:
                if is_valid:
                    st.info("Click the 'Predict Salary' button on the left to calculate the expected salary range.")

# ----------------- MODEL PERFORMANCE PAGE -----------------
elif menu_option == "📈 Model Performance & Tuning":
    st.markdown('<div class="section-header">Model Training, Tuning & Evaluation</div>', unsafe_allow_html=True)
    
    if metadata is not None:
        metrics_list = metadata.get("metrics", [])
        df_metrics = pd.DataFrame(metrics_list)
        
        # Display comparison table
        st.write("### Model Performance Comparison (Sorted by R² Score)")
        df_metrics_sorted = df_metrics.sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
        
        # Format table columns
        formatted_df = df_metrics_sorted.copy()
        formatted_df["MAE"] = formatted_df["MAE"].map("₹{:,.2f}".format)
        formatted_df["MSE"] = formatted_df["MSE"].map("{:,.2f}".format)
        formatted_df["RMSE"] = formatted_df["RMSE"].map("₹{:,.2f}".format)
        formatted_df["R2 Score"] = formatted_df["R2 Score"].map("{:.4f}".format)
        
        # Map CV columns if available
        if "CV R2 Mean" in formatted_df.columns:
            formatted_df["CV R2 Score"] = formatted_df["CV R2 Mean"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
            formatted_df = formatted_df.drop(columns=["CV R2 Mean", "CV R2 Std", "MSE"], errors="ignore")
            # Reorder columns
            cols_order = ["Model", "MAE", "RMSE", "R2 Score", "CV R2 Score"]
            formatted_df = formatted_df[[c for c in cols_order if c in formatted_df.columns]]
            
        st.dataframe(formatted_df, use_container_width=True)
        
        # Page tabs
        tab_charts, tab_tuning, tab_plots, tab_imp = st.tabs([
            "📊 Comparison Charts", 
            "⚙️ Hyperparameter Tuning", 
            "📈 Diagnostic Plots & Learning Curves",
            "🧬 Feature Importance & Insights"
        ])
        
        with tab_charts:
            # Plots comparing R2 Score & MAE
            col1, col2 = st.columns(2)
            with col1:
                fig_bar_r2 = px.bar(
                    df_metrics_sorted, 
                    x="Model", 
                    y="R2 Score", 
                    color="Model",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title="R² Score Comparison (Higher is Better)",
                    text_auto=".4f"
                )
                st.plotly_chart(fig_bar_r2, use_container_width=True)
            with col2:
                fig_bar_mae = px.bar(
                    df_metrics_sorted, 
                    x="Model", 
                    y="MAE", 
                    color="Model",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title="MAE Comparison (Lower is Better)",
                    text_auto=".1f"
                )
                st.plotly_chart(fig_bar_mae, use_container_width=True)
                
        with tab_tuning:
            # Hyperparameter tuning section
            st.markdown('<div class="section-subheader" style="font-size:1.2rem;font-weight:700;margin-bottom:10px;">Hyperparameter Tuning (Random Forest)</div>', unsafe_allow_html=True)
            tuning = metadata.get("tuning", {})
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("#### GridSearchCV Summary")
                st.write(f"- **Grid Search Duration:** {tuning.get('grid_time', 0.0):.2f} seconds")
                st.write(f"- **Best R² Score (CV):** {tuning.get('grid_best_score', 0.0):.4f}")
                st.write("**Best Parameters Found:**")
                st.json(tuning.get("grid_best_params", {}))
                
            with col_t2:
                st.markdown("#### RandomizedSearchCV Summary")
                st.write(f"- **Random Search Duration:** {tuning.get('random_time', 0.0):.2f} seconds")
                st.write(f"- **Best R² Score (CV):** {tuning.get('random_best_score', 0.0):.4f}")
                st.write("**Best Parameters Found:**")
                st.json(tuning.get("random_best_params", {}))
                
            st.markdown("#### Before vs After Tuning Performance Comparison (on Test Set)")
            # Get RF Untuned and RF Tuned rows
            rf_row = df_metrics_sorted[df_metrics_sorted["Model"] == "Random Forest"]
            trf_row = df_metrics_sorted[df_metrics_sorted["Model"] == "Tuned Random Forest"]
            
            if not rf_row.empty and not trf_row.empty:
                comp_data = {
                    "Metric": ["R2 Score", "MAE", "RMSE"],
                    "Before Tuning (Base RF)": [rf_row.iloc[0]["R2 Score"], rf_row.iloc[0]["MAE"], rf_row.iloc[0]["RMSE"]],
                    "After Tuning (Tuned RF)": [trf_row.iloc[0]["R2 Score"], trf_row.iloc[0]["MAE"], trf_row.iloc[0]["RMSE"]]
                }
                comp_df = pd.DataFrame(comp_data)
                # Format
                comp_df["Before Tuning (Base RF)"] = comp_df["Before Tuning (Base RF)"].map(lambda x: f"{x:.4f}" if x <= 1 else f"₹{x:,.2f}")
                comp_df["After Tuning (Tuned RF)"] = comp_df["After Tuning (Tuned RF)"].map(lambda x: f"{x:.4f}" if x <= 1 else f"₹{x:,.2f}")
                st.table(comp_df)
            else:
                st.write("Random Forest performance metrics missing from comparison.")
                
            st.markdown(
                "💡 **Tuning Interpretation**: RandomizedSearchCV completed significantly faster "
                "while achieving virtually the same R² score as GridSearchCV. "
                "Tuning restricted Random Forest tree depth (`max_depth: 10`) to reduce overfitting, "
                "leading to a robust and generalizable model."
            )
            
        with tab_plots:
            st.markdown("#### Model Diagnostic & Evaluation Charts")
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                actual_pred_path = os.path.join(BASE_DIR, "reports", "actual_vs_predicted.png")
                if os.path.exists(actual_pred_path):
                    st.image(actual_pred_path, caption="Prediction Error Plot (Actual vs Predicted)", use_column_width=True)
                else:
                    st.info("Actual vs Predicted plot not found.")
            with col_img2:
                residuals_path = os.path.join(BASE_DIR, "reports", "residuals_plot.png")
                if os.path.exists(residuals_path):
                    st.image(residuals_path, caption="Residual Plot (Residuals vs Predicted)", use_column_width=True)
                else:
                    st.info("Residual plot not found.")
                    
            col_img3, col_img4 = st.columns(2)
            with col_img3:
                error_dist_path = os.path.join(BASE_DIR, "reports", "error_distribution.png")
                if os.path.exists(error_dist_path):
                    st.image(error_dist_path, caption="Residual Error Distribution (Normality Check)", use_column_width=True)
                else:
                    st.info("Error distribution plot not found.")
            with col_img4:
                learning_curve_path = os.path.join(BASE_DIR, "reports", "learning_curve.png")
                if os.path.exists(learning_curve_path):
                    st.image(learning_curve_path, caption="Model Learning Curve (Train vs Validation R²)", use_column_width=True)
                else:
                    st.info("Learning curve plot not found.")
                    
        with tab_imp:
            # Feature Importance section
            st.write("### Feature Importance Analysis")
            imp_list = metadata.get("rf_importance", [])
            if imp_list:
                df_imp = pd.DataFrame(imp_list)
                fig_imp = px.bar(
                    df_imp, 
                    x="Importance", 
                    y="Feature", 
                    orientation="h",
                    title="Random Forest: Relative Feature Importance",
                    color="Feature",
                    color_discrete_sequence=["#10B981", "#3B82F6"],
                    text_auto=".4f"
                )
                fig_imp.update_layout(xaxis=dict(range=[0, 1.1]))
                st.plotly_chart(fig_imp, use_container_width=True)
                
                st.markdown(
                    "#### Interpretability Insights:\n"
                    "- **Years of Experience** is the most significant driver for salary prediction, carrying over **96.96%** of the relative predictive weight.\n"
                    "- **Age** contributes less than **3.04%** of the weight. This indicates that once experience is controlled for, age itself has very little impact on salary.\n"
                    "- Linear models (Linear, Ridge, Lasso) and Non-linear models (Decision Tree, Random Forest, XGBoost) perform extremely closely, showing that the relationship in this dataset is highly linear with structured noise."
                )
    else:
        st.warning("Model metadata is missing. Please run `run_pipeline.py` to train models and generate metrics.")

# ----------------- ABOUT PROJECT PAGE -----------------
elif menu_option == "ℹ️ About Project":
    st.markdown('<div class="section-header">About the HR Salary Predictor Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview & Objective
    This project is an end-to-end Machine Learning pipeline and Streamlit dashboard built to predict and benchmark the salaries of HR professionals based on objective metrics.
    
    In corporate environments, standardizing compensation structures helps maintain pay equity, improve employee satisfaction, and speed up recruitment processes.
    
    ### 📊 Dataset Details
    - **Total Records**: 200,000 rows (Raw)
    - **Features**: 
      - `Age` (Integer: Age of employee from 22 to 59 years)
      - `Years_of_Experience` (Integer: Experience in years from 0 to 39 years)
    - **Target Variable**: `Target_Salary` (Decimal: Annual salary in INR)
    
    ### 🧼 Data Quality & Cleaning Strategy
    To ensure data integrity, the pipeline implements strict validation:
    - **Data Type Casting**: Forces columns into proper numeric representations.
    - **Negative Salaries**: Filtered out physically impossible negative salaries (~1% of records).
    - **Logical Inconsistencies**: Dropped rows where experience exceeds age limitations (`Experience > Age - 18`), representing ~42% of the synthetically generated noise.
    - **Outlier Detection**: Performed IQR validation on target salary, deciding to keep valid high compensation ranges to avoid skewing predictions.
    
    ### ⚙️ Feature Engineering
    - **Feature Scaling**: Scaled age and experience using `StandardScaler` to bring them to identical scales, which is critical for regularized models (Lasso, Ridge).
    - **Feature Selection**: Handled multicollinearity checks (low correlation between Age and Experience).
    
    ### 🔮 Modeling and Tuning
    We trained and cross-validated 6 different regression algorithms:
    1. **Linear Regression** (Baseline)
    2. **Ridge Regression** (L2 Regularization)
    3. **Lasso Regression** (L1 Regularization - Selected Best Model)
    4. **Decision Tree Regressor**
    5. **Random Forest Regressor** (Ensemble)
    6. **XGBoost Regressor** (Gradient Boosting)
    
    Hyperparameters for Random Forest were tuned using **GridSearchCV** and **RandomizedSearchCV** to restrict tree depth and prevent overfitting.
    
    ### 📈 Evaluation Metrics
    - **MAE (Mean Absolute Error)**: Measures the average magnitude of errors (approx. ₹3,889 INR).
    - **RMSE (Root Mean Squared Error)**: Standard deviation of residuals (approx. ₹4,846 INR).
    - **R² Score**: Captures the percentage of variance explained by features (~53.86%).
    - **5-Fold Cross-Validation**: Validates model stability across training folds (~54.36%).
    """)

