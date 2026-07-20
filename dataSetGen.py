import pandas as pd
import numpy as np
import os

np.random.seed(42)

n_samples = 100000

# Feature definitions & options
job_titles = [
    'Software Engineer', 'Senior Software Engineer', 'Lead Data Scientist', 
    'Engineering Manager', 'HR Specialist', 'HR Manager', 
    'Financial Analyst', 'Marketing Specialist', 'Sales Executive', 'Product Manager'
]

departments = [
    'Engineering', 'Data & Analytics', 'Human Resources', 
    'Finance', 'Sales & Marketing', 'Product'
]

education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
location_tiers = ['Tier 1', 'Tier 2', 'Tier 3 / Remote']

# Base salary additions by Role ($)
role_base_salary = {
    'Software Engineer': 30000,
    'Senior Software Engineer': 55000,
    'Lead Data Scientist': 75000,
    'Engineering Manager': 80000,
    'HR Specialist': 20000,
    'HR Manager': 45000,
    'Financial Analyst': 35000,
    'Marketing Specialist': 22000,
    'Sales Executive': 25000,
    'Product Manager': 60000
}

# Education additions ($)
education_bonus = {
    "High School": 0,
    "Bachelor's": 12000,
    "Master's": 25000,
    "PhD": 42000
}

# Location multipliers
location_multiplier = {
    'Tier 1': 1.25,
    'Tier 2': 1.05,
    'Tier 3 / Remote': 0.90
}

# Generate random samples
age = np.random.randint(22, 63, n_samples)
# Ensure Experience <= Age - 20 logically
max_exp = np.maximum(0, age - 20)
years_of_experience = np.array([np.random.randint(0, m + 1) for m in max_exp])

job_title_arr = np.random.choice(job_titles, n_samples)
dept_arr = np.random.choice(departments, n_samples)
edu_arr = np.random.choice(education_levels, n_samples, p=[0.1, 0.5, 0.3, 0.1])
loc_arr = np.random.choice(location_tiers, n_samples, p=[0.4, 0.4, 0.2])
perf_rating = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.50, 0.20, 0.10])
certifications = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.3, 0.3, 0.2, 0.1, 0.07, 0.03])

# Calculate Base Salary
base_salary = 30000

role_comp = np.array([role_base_salary[r] for r in job_title_arr])
edu_comp = np.array([education_bonus[e] for e in edu_arr])
loc_mult = np.array([location_multiplier[l] for l in loc_arr])
exp_comp = years_of_experience * 2800
cert_comp = certifications * 1800
perf_mult = 1.0 + (perf_rating - 3) * 0.06

# Final Target Salary
target_salary = (base_salary + role_comp + edu_comp + exp_comp + cert_comp) * loc_mult * perf_mult
noise = np.random.normal(0, 3500, n_samples)
target_salary = np.round(target_salary + noise, 2)
# Ensure no negative salaries
target_salary = np.maximum(20000, target_salary)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Years_of_Experience': years_of_experience,
    'Job_Title': job_title_arr,
    'Department': dept_arr,
    'Education_Level': edu_arr,
    'Location_Tier': loc_arr,
    'Performance_Rating': perf_rating,
    'Certifications': certifications,
    'Target_Salary': target_salary
})

# Save to data/raw/hr_salary_data.csv
os.makedirs("data/raw", exist_ok=True)
output_path = "data/raw/hr_salary_data.csv"
df.to_csv(output_path, index=False)
print(f"Generated {n_samples} realistic records with 8 features!")
print(f"Saved dataset to {output_path}")
print(df.head())

