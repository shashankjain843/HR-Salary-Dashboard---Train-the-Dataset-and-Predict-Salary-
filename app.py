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
    page_title="HR Salary Prediction & Compensation Benchmarking System",
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

@st.cache_data
def load_raw_data(file_path):
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

# Load model and preprocessor
@st.cache_resource
def load_model_artifacts(model_path, prep_path):
    model = None
    preprocessor = None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(prep_path):
        preprocessor = joblib.load(prep_path)
    return model, preprocessor

# Define file paths
BASE_DIR = r"c:\Users\Shashank\OneDrive\ドキュメント\hr_sal_dashboard"
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "hr_salary_cleaned.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "hr_salary_data.csv")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# Load assets
df_cleaned = load_processed_data(CLEANED_DATA_PATH)
df_raw = load_raw_data(RAW_DATA_PATH)
metadata = load_metadata(METADATA_PATH)
model, preprocessor = load_model_artifacts(MODEL_PATH, PREPROCESSOR_PATH)

# Feature choices for selectboxes
JOB_TITLES = [
    'Software Engineer', 'Senior Software Engineer', 'Lead Data Scientist', 
    'Engineering Manager', 'HR Specialist', 'HR Manager', 
    'Financial Analyst', 'Marketing Specialist', 'Sales Executive', 'Product Manager'
]
DEPARTMENTS = ['Engineering', 'Data & Analytics', 'Human Resources', 'Finance', 'Sales & Marketing', 'Product']
EDUCATION_LEVELS = ["High School", "Bachelor's", "Master's", "PhD"]
LOCATION_TIERS = ['Tier 1', 'Tier 2', 'Tier 3 / Remote']

# Sidebar Navigation
st.sidebar.markdown("### 💼 HR Salary Analytics & ML")
st.sidebar.markdown("---")

menu_option = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home & Overview",
        "📊 Data Explorer & EDA",
        "🔮 Single Salary Predictor",
        "📂 Batch Salary Predictor",
        "📈 Model Performance & Tuning",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

if metadata:
    st.sidebar.markdown(f"**Best Model:** `{metadata.get('best_model_name', 'Tuned Random Forest')}`")
    st.sidebar.markdown(f"**Accuracy (R² Score):** `98.86%`")
    st.sidebar.markdown(f"**Target Variable:** `Target_Salary` (INR)")
    st.sidebar.markdown("**Features Count:** `8 Features`")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='sidebar-text'>Enterprise-grade compensation benchmarking engine using multi-model Machine Learning.</div>", 
    unsafe_allow_html=True
)

# Header Section
st.markdown('<div class="main-header">HR Compensation Benchmarking & Salary Intelligence</div>', unsafe_allow_html=True)

# ----------------- HOME & OVERVIEW PAGE -----------------
if menu_option == "🏠 Home & Overview":
    st.markdown('<div class="section-header">Project Overview & Objectives</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(
            "This platform utilizes historical payroll datasets to build robust, explainable predictive models "
            "for HR salary benchmarking. Incorporating **8 key professional parameters** (Role, Department, Education, "
            "Location Tier, Performance Rating, Certifications, Age, and Experience), the engine predicts accurate compensation packages."
        )
        st.markdown("""
        ### Core Business Objectives:
        - **Fair Compensation Benchmarking**: Standardize compensation based on role, education, location, and performance.
        - **Data Leakage & Inconsistency Protection**: Filter out illogical data entries where work experience exceeds physical age constraints.
        - **High Precision ML Pipeline**: Compare baseline regression models with decision ensembles (Random Forest, XGBoost) using One-Hot Encoding and Scaling.
        - **Enterprise Model Accuracy**: Achieved **> 98.8% R² Score** on test benchmarks.
        """)
        
    with col2:
        st.info(
            "💡 **Model Performance Summary**:\n\n"
            "- **Best Model**: Tuned Random Forest\n"
            "- **R² Score**: **0.9886** (98.86% Accuracy)\n"
            "- **MAE**: **₹3,667.70 INR**\n"
            "- **Cross-Validation R²**: **0.9878**"
        )

    # KPI Metrics Section
    st.markdown('<div class="section-header">Dashboard KPIs (Cleaned Payroll Data)</div>', unsafe_allow_html=True)
    
    if df_cleaned is not None:
        num_records = len(df_cleaned)
        avg_salary = df_cleaned["Target_Salary"].mean()
        max_salary = df_cleaned["Target_Salary"].max()
        min_salary = df_cleaned["Target_Salary"].min()
        avg_experience = df_cleaned["Years_of_Experience"].mean()
        
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
            st.metric(label="Total Job Titles", value=f"{df_cleaned['Job_Title'].nunique()}")
    else:
        st.warning("Processed dataset not found! Please run the pipeline script `run_pipeline.py` first.")

    # Dataset Preview Tab
    st.markdown('<div class="section-header">Dataset Preview & Schema Documentation</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📄 First 10 Sample Rows", "📊 Column Definitions & Types"])
    
    with tab1:
        if df_cleaned is not None:
            st.dataframe(df_cleaned.head(10), use_container_width=True)
            st.caption(f"Showing first 10 rows. Cleaned dataset dimensions: **{df_cleaned.shape[0]} rows x {df_cleaned.shape[1]} columns**")
            
    with tab2:
        schema_df = pd.DataFrame({
            "Column Name": ["Age", "Years_of_Experience", "Job_Title", "Department", "Education_Level", "Location_Tier", "Performance_Rating", "Certifications", "Target_Salary"],
            "Data Type": ["int64", "int64", "object (Categorical)", "object (Categorical)", "object (Categorical)", "object (Categorical)", "int64", "int64", "float64"],
            "Role": ["Numerical Feature", "Numerical Feature", "Categorical Feature", "Categorical Feature", "Categorical Feature", "Categorical Feature", "Numerical Feature", "Numerical Feature", "Target Variable"],
            "Description": [
                "Age of the employee (22 to 62 years).",
                "Total work experience (0 to 40 years, Exp <= Age - 18).",
                "Designation / Role (Software Engineer, Lead Data Scientist, etc.).",
                "Department (Engineering, Finance, HR, Sales, etc.).",
                "Highest Education Qualification (High School, Bachelor's, Master's, PhD).",
                "City Tier / Cost of living factor (Tier 1, Tier 2, Tier 3 / Remote).",
                "Performance rating score on scale 1 to 5.",
                "Number of professional certifications held (0 to 5).",
                "Annual compensation package in INR."
            ]
        })
        st.table(schema_df)

# ----------------- DATA EXPLORER & EDA PAGE -----------------
elif menu_option == "📊 Data Explorer & EDA":
    st.markdown('<div class="section-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    
    if df_cleaned is not None:
        dist_tab, cat_tab, corr_tab, rel_tab, outlier_tab = st.tabs([
            "📈 Numerical Distributions", 
            "🏢 Categorical & Salary Breakdown",
            "🔥 Feature Relationship & Correlation", 
            "🔗 Interactive Scatter Matrix", 
            "🚫 Outlier Analysis"
        ])
        
        with dist_tab:
            st.write("### Numerical Feature Distributions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fig_age = px.histogram(df_cleaned, x="Age", nbins=20, title="Age Distribution", color_discrete_sequence=["#3B82F6"])
                st.plotly_chart(fig_age, use_container_width=True)
                
            with col2:
                fig_exp = px.histogram(df_cleaned, x="Years_of_Experience", nbins=20, title="Experience Distribution", color_discrete_sequence=["#10B981"])
                st.plotly_chart(fig_exp, use_container_width=True)
                
            with col3:
                fig_perf = px.histogram(df_cleaned, x="Performance_Rating", title="Performance Rating Distribution", color_discrete_sequence=["#F59E0B"])
                st.plotly_chart(fig_perf, use_container_width=True)
                
            with col4:
                fig_cert = px.histogram(df_cleaned, x="Certifications", title="Certifications Count", color_discrete_sequence=["#8B5CF6"])
                st.plotly_chart(fig_cert, use_container_width=True)
                
        with cat_tab:
            st.write("### Average Salary by Role, Education & Location")
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                df_role = df_cleaned.groupby("Job_Title")["Target_Salary"].mean().reset_index().sort_values(by="Target_Salary", ascending=True)
                fig_role = px.bar(df_role, x="Target_Salary", y="Job_Title", orientation="h", title="Average Salary by Job Title", color="Target_Salary", color_continuous_scale="Blues")
                st.plotly_chart(fig_role, use_container_width=True)
                
            with col_c2:
                df_edu = df_cleaned.groupby("Education_Level")["Target_Salary"].mean().reset_index().sort_values(by="Target_Salary", ascending=True)
                fig_edu = px.bar(df_edu, x="Education_Level", y="Target_Salary", title="Average Salary by Education Level", color="Education_Level", color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_edu, use_container_width=True)
                
            col_c3, col_c4 = st.columns(2)
            with col_c3:
                df_loc = df_cleaned.groupby("Location_Tier")["Target_Salary"].mean().reset_index()
                fig_loc = px.pie(df_loc, values="Target_Salary", names="Location_Tier", title="Salary Weight Distribution by Location Tier", color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_loc, use_container_width=True)
                
            with col_c4:
                df_dept = df_cleaned.groupby("Department")["Target_Salary"].mean().reset_index().sort_values(by="Target_Salary", ascending=False)
                fig_dept = px.bar(df_dept, x="Department", y="Target_Salary", title="Average Salary by Department", color="Department", color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig_dept, use_container_width=True)

        with corr_tab:
            st.write("### Numerical Correlations & Salary Drivers")
            num_df = df_cleaned[["Age", "Years_of_Experience", "Performance_Rating", "Certifications", "Target_Salary"]]
            corr = num_df.corr()
            
            fig_heat = px.imshow(
                corr, text_auto=".3f", aspect="auto", color_continuous_scale="RdBu_r",
                title="Numerical Correlation Heatmap Matrix", zmin=-1, zmax=1
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with rel_tab:
            st.write("### Experience vs Salary (Colored by Job Title)")
            sample_size = st.slider("Select sample size for scatter plot", 1000, 10000, 3000, step=1000)
            df_sample = df_cleaned.sample(sample_size, random_state=42)
            
            fig_scatter = px.scatter(
                df_sample, x="Years_of_Experience", y="Target_Salary", color="Job_Title",
                title=f"Experience vs Target Salary across Roles (Sample of {sample_size:,} records)", opacity=0.7
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with outlier_tab:
            st.write("### Outlier Detection (Target Salary)")
            fig_box = px.box(df_cleaned, x="Job_Title", y="Target_Salary", color="Job_Title", title="Salary Spread and Outliers by Job Designation")
            st.plotly_chart(fig_box, use_container_width=True)

# ----------------- SINGLE PREDICTOR PAGE -----------------
elif menu_option == "🔮 Single Salary Predictor":
    st.markdown('<div class="section-header">Interactive Compensation Benchmarking Calculator</div>', unsafe_allow_html=True)
    
    if model is None or preprocessor is None:
        st.error("❌ Trained model or preprocessor could not be loaded. Please run `run_pipeline.py` first.")
    else:
        st.write("Fill in the candidate/employee profile details to generate an instant ML salary recommendation:")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Input Professional Profile")
            job_title = st.selectbox("Job Designation / Title:", JOB_TITLES)
            department = st.selectbox("Department:", DEPARTMENTS)
            education_level = st.selectbox("Education Level:", EDUCATION_LEVELS, index=1)
            location_tier = st.selectbox("Location Tier / Cost of Living:", LOCATION_TIERS, index=0)
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                age = st.number_input("Age (Years):", min_value=18, max_value=65, value=30, step=1)
                performance_rating = st.slider("Performance Rating (1-5):", min_value=1, max_value=5, value=3, step=1)
            with sub_col2:
                years_experience = st.number_input("Years of Experience:", min_value=0, max_value=40, value=5, step=1)
                certifications = st.slider("Certifications Count:", min_value=0, max_value=5, value=1, step=1)
                
            # Input validations
            is_valid = True
            if years_experience > (age - 18):
                st.warning(
                    f"⚠️ **Physical Inconsistency Warning**: Aged **{age}** cannot have **{years_experience}** years of experience. "
                    f"(Started working at age {age - years_experience}, younger than 18)."
                )

        with col2:
            st.markdown("### Prediction Benchmark")
            
            predict_btn = st.button("Calculate Salary Benchmark", use_container_width=True, type="primary")
            
            if predict_btn:
                try:
                    profile_df = pd.DataFrame([{
                        "Age": age,
                        "Years_of_Experience": years_experience,
                        "Job_Title": job_title,
                        "Department": department,
                        "Education_Level": education_level,
                        "Location_Tier": location_tier,
                        "Performance_Rating": performance_rating,
                        "Certifications": certifications
                    }])
                    
                    transformed_features = preprocessor.transform(profile_df)
                    predicted_salary = max(20000.0, float(model.predict(transformed_features)[0]))
                    
                    st.markdown(
                        f'<div class="prediction-box">'
                        f'<p style="margin:0; font-size:1rem; color:#4B5563; font-weight:600;">RECOMMENDED SALARY PACKAGE</p>'
                        f'<p class="prediction-value">₹{predicted_salary:,.2f}</p>'
                        f'<p style="margin:0; font-size:0.85rem; color:#059669; font-weight:600;">'
                        f'✅ Benchmarked with 98.86% Accuracy ({metadata.get("best_model_name", "Tuned Random Forest") if metadata else "ML Model"})</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    with st.expander("🔍 Benchmark Breakdown & Profile Analysis", expanded=True):
                        st.write(f"- **Designation:** {job_title} ({department})")
                        st.write(f"- **Qualification:** {education_level} Degree")
                        st.write(f"- **Location & Exp:** {location_tier} | {years_experience} Years Exp (Age {age})")
                        st.write(f"- **Performance & Certs:** Rating {performance_rating}/5 | {certifications} Certifications")
                        st.write(f"- **Model Margin of Error (MAE):** ± ₹3,667.70 INR")
                except Exception as ex:
                    st.error(f"Prediction failed: {str(ex)}")

# ----------------- BATCH PREDICTOR PAGE -----------------
elif menu_option == "📂 Batch Salary Predictor":
    st.markdown('<div class="section-header">Batch Employee Compensation Predictor (CSV)</div>', unsafe_allow_html=True)
    
    st.write("Upload a CSV containing multiple employee profiles to calculate batch salary recommendations at scale:")
    
    # Template Download
    sample_df = pd.DataFrame([{
        "Age": 28, "Years_of_Experience": 5, "Job_Title": "Software Engineer", 
        "Department": "Engineering", "Education_Level": "Bachelor's", 
        "Location_Tier": "Tier 1", "Performance_Rating": 4, "Certifications": 2
    }, {
        "Age": 42, "Years_of_Experience": 18, "Job_Title": "Lead Data Scientist", 
        "Department": "Data & Analytics", "Education_Level": "Master's", 
        "Location_Tier": "Tier 1", "Performance_Rating": 5, "Certifications": 4
    }])
    
    csv_template = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample Batch Template (CSV)",
        data=csv_template,
        file_name="batch_salary_prediction_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file to process", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Batch Preview (First 5 rows):")
            st.dataframe(batch_df.head(5), use_container_width=True)
            
            required_cols = ["Age", "Years_of_Experience", "Job_Title", "Department", "Education_Level", "Location_Tier", "Performance_Rating", "Certifications"]
            missing_cols = [c for c in required_cols if c not in batch_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns in uploaded CSV: `{missing_cols}`")
            else:
                if st.button("🚀 Process Batch Salary Predictions", type="primary"):
                    with st.spinner("Processing predictions..."):
                        batch_features = batch_df[required_cols]
                        trans_batch = preprocessor.transform(batch_features)
                        preds = model.predict(trans_batch)
                        
                        batch_df["Predicted_Salary_INR"] = np.round(np.maximum(20000.0, preds), 2)
                        
                        st.success(f"✅ Successfully processed {len(batch_df):,} employee records!")
                        st.dataframe(batch_df, use_container_width=True)
                        
                        out_csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predicted Salaries CSV",
                            data=out_csv,
                            file_name="batch_predicted_salaries.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Failed to process CSV file: {str(e)}")

# ----------------- MODEL PERFORMANCE PAGE -----------------
elif menu_option == "📈 Model Performance & Tuning":
    st.markdown('<div class="section-header">Model Training, Tuning & Evaluation</div>', unsafe_allow_html=True)
    
    if metadata is not None:
        metrics_list = metadata.get("metrics", [])
        df_metrics = pd.DataFrame(metrics_list)
        
        st.write("### Model Performance Comparison Table (Sorted by R² Score)")
        df_metrics_sorted = df_metrics.sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
        
        formatted_df = df_metrics_sorted.copy()
        formatted_df["MAE"] = formatted_df["MAE"].map("₹{:,.2f}".format)
        formatted_df["RMSE"] = formatted_df["RMSE"].map("₹{:,.2f}".format)
        formatted_df["R2 Score"] = formatted_df["R2 Score"].map("{:.4f}".format)
        if "CV R2 Mean" in formatted_df.columns:
            formatted_df["CV R2 Score"] = formatted_df["CV R2 Mean"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
            formatted_df = formatted_df.drop(columns=["CV R2 Mean", "CV R2 Std", "MSE"], errors="ignore")
            cols_order = ["Model", "MAE", "RMSE", "R2 Score", "CV R2 Score"]
            formatted_df = formatted_df[[c for c in cols_order if c in formatted_df.columns]]
            
        st.dataframe(formatted_df, use_container_width=True)
        
        tab_charts, tab_tuning, tab_plots, tab_imp = st.tabs([
            "📊 Comparison Charts", 
            "⚙️ Hyperparameter Tuning", 
            "📈 Diagnostic Plots",
            "🧬 Feature Importance"
        ])
        
        with tab_charts:
            col1, col2 = st.columns(2)
            with col1:
                fig_bar_r2 = px.bar(
                    df_metrics_sorted, x="Model", y="R2 Score", color="Model",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title="R² Score Comparison (Higher is Better)", text_auto=".4f"
                )
                fig_bar_r2.update_layout(yaxis=dict(range=[0.9, 1.0]))
                st.plotly_chart(fig_bar_r2, use_container_width=True)
            with col2:
                fig_bar_mae = px.bar(
                    df_metrics_sorted, x="Model", y="MAE", color="Model",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title="MAE Comparison (Lower is Better)", text_auto=".1f"
                )
                st.plotly_chart(fig_bar_mae, use_container_width=True)
                
        with tab_tuning:
            st.markdown("#### Hyperparameter Tuning Summary (Random Forest)")
            tuning = metadata.get("tuning", {})
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("##### GridSearchCV")
                st.write(f"- Duration: {tuning.get('grid_time', 0.0):.2f} sec")
                st.write(f"- Best CV R² Score: {tuning.get('grid_best_score', 0.0):.4f}")
                st.json(tuning.get("grid_best_params", {}))
            with col_t2:
                st.markdown("##### RandomizedSearchCV")
                st.write(f"- Duration: {tuning.get('random_time', 0.0):.2f} sec")
                st.write(f"- Best CV R² Score: {tuning.get('random_best_score', 0.0):.4f}")
                st.json(tuning.get("random_best_params", {}))
                
        with tab_plots:
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                actual_pred_path = os.path.join(BASE_DIR, "reports", "actual_vs_predicted.png")
                if os.path.exists(actual_pred_path):
                    st.image(actual_pred_path, caption="Actual vs Predicted Salary", use_column_width=True)
            with col_img2:
                residuals_path = os.path.join(BASE_DIR, "reports", "residuals_plot.png")
                if os.path.exists(residuals_path):
                    st.image(residuals_path, caption="Residual Analysis Plot", use_column_width=True)
                    
        with tab_imp:
            st.write("### Relative Feature Importance (Transformed Schema)")
            imp_list = metadata.get("rf_importance", [])
            if imp_list:
                df_imp = pd.DataFrame(imp_list).head(15)
                fig_imp = px.bar(
                    df_imp, x="Importance", y="Feature", orientation="h",
                    title="Top 15 Transformed Features by Relative Importance",
                    color="Feature", color_discrete_sequence=px.colors.qualitative.Vivid, text_auto=".4f"
                )
                st.plotly_chart(fig_imp, use_container_width=True)

# ----------------- ABOUT PROJECT PAGE -----------------
elif menu_option == "ℹ️ About Project":
    st.markdown('<div class="section-header">About the HR Salary Predictor Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    An enterprise Machine Learning solution for automating HR salary benchmarking and pay equity evaluation.
    
    ### 📊 Features Schema (8 Parameters)
    1. `Job_Title` (Categorical: 10 Roles)
    2. `Department` (Categorical: 6 Departments)
    3. `Education_Level` (Categorical: High School to PhD)
    4. `Location_Tier` (Categorical: Tier 1, Tier 2, Tier 3)
    5. `Performance_Rating` (Numerical: 1 to 5)
    6. `Certifications` (Numerical: 0 to 5)
    7. `Age` (Numerical: 22 to 62 years)
    8. `Years_of_Experience` (Numerical: 0 to 40 years, Exp <= Age - 18)
    
    ### 🚀 ML Pipeline Architecture
    - **Preprocessing**: `ColumnTransformer` with `StandardScaler` for numeric columns & `OneHotEncoder` for categorical columns.
    - **Models Trained**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost.
    - **Evaluation Metrics**: $R^2$ Score (**98.86%**), MAE (**₹3,667.70**), 5-Fold Cross Validation.
    """)
