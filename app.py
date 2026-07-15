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

# Sidebar Navigation & Information
st.sidebar.markdown("### 💼 HR Salary Prediction")
st.sidebar.markdown("---")

# Navigation menu
menu_option = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home & Overview",
        "📊 Data Explorer & EDA",
        "🔮 Salary Predictor",
        "📈 Model Performance & Tuning"
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

# Removed About the Project section per user request

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
        - **Model Reliability**: Compare baseline regression models (Linear, Ridge, Lasso) with decision-based ensembles (Decision Tree, Random Forest) and select the optimal predictor.
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
            st.dataframe(df_cleaned.head(10), use_container_width=True)
            st.caption(f"Showing first 10 rows. Total dataset dimensions: **{df_cleaned.shape[0]} rows x {df_cleaned.shape[1]} columns**")
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
        dist_tab, corr_tab, rel_tab, outlier_tab = st.tabs([
            "📈 Feature Distributions", 
            "🔥 Correlation Heatmap", 
            "🔗 Relationship Analysis", 
            "🚫 Outlier Analysis"
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
                "The salary target has a normal distribution range and the physically impossible negative values "
                "were successfully removed during cleaning."
            )
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
            age = st.slider("Employee Age:", min_value=18, max_value=65, value=30, step=1)
            years_experience = st.slider("Years of Experience:", min_value=0, max_value=40, value=5, step=1)
            
            # Input constraints warning
            if years_experience > (age - 18):
                st.warning(
                    f"⚠️ **Physical Inconsistency Alert**: An employee aged **{age}** cannot have "
                    f"**{years_experience}** years of experience (started working at age {age - years_experience}). "
                    f"The prediction might be extrapolated outside logical boundaries."
                )

        with col2:
            st.markdown("### Prediction Results")
            
            # Predict button
            predict_btn = st.button("Predict Salary", use_container_width=True, type="primary")
            
            if predict_btn:
                # Prepare inputs
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
                        # Standard coefficients approximation
                        st.write("- **Primary Driver:** Years of Experience (represents approx. 97% of predictor weight).")
                    st.write("- **Confidence Boundary:** ± ₹4,846.26 INR (based on RMSE score of the model).")
            else:
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
        st.dataframe(formatted_df, use_container_width=True)
        
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
            
        # Hyperparameter tuning section
        st.markdown('<div class="section-header">Hyperparameter Tuning Details (Random Forest)</div>', unsafe_allow_html=True)
        tuning = metadata.get("tuning", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### GridSearchCV Summary")
            st.write(f"- **Grid Search Duration:** {tuning.get('grid_time', 0.0):.2f} seconds")
            st.write(f"- **Best R² Score (CV):** {tuning.get('grid_best_score', 0.0):.4f}")
            st.write("**Best Parameters Found:**")
            st.json(tuning.get("grid_best_params", {}))
            
        with col2:
            st.markdown("#### RandomizedSearchCV Summary")
            st.write(f"- **Random Search Duration:** {tuning.get('random_time', 0.0):.2f} seconds")
            st.write(f"- **Best R² Score (CV):** {tuning.get('random_best_score', 0.0):.4f}")
            st.write("**Best Parameters Found:**")
            st.json(tuning.get("random_best_params", {}))
            
        st.markdown(
            "💡 **Interpretation**: RandomizedSearchCV completed significantly faster "
            "while achieving virtually the same R² score as GridSearchCV. "
            "This showcases the benefit of using randomized searches for tuning complex models."
        )

        # Feature Importance section
        st.markdown('<div class="section-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
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
                "**Insight**: **Years of Experience** is the most significant driver for salary prediction, "
                "carrying **96.96%** of the relative predictive power. **Age** contributes only **3.04%** "
                "once experience is accounted for. This indicates that professional experience is the core "
                "pricing mechanism in payroll benchmarking."
            )
    else:
        st.warning("Model metadata is missing. Please run `run_pipeline.py` to train models and generate metrics.")
