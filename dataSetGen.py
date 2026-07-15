import pandas as pd
import numpy as np
from faker import Faker

# Initialize Faker and random seed
faker = Faker()
np.random.seed(42)

# Define number of samples for the dataset
n_samples = 200000

# Generate features for the dataset
age = np.random.randint(22, 60, n_samples)
years_of_experience = np.random.randint(0, 40, n_samples)

# Create a basic linear model for the salary based on Age and Years_of_Experience
# Example model: Salary = 3000 + (Age * 100) + (Years_of_Experience * 500)
base_salary = 3000 + (age * 100) + (years_of_experience * 500)

# Add noise to simulate real-world salary variation
salary_noise = np.random.normal(0, 5000, n_samples)  # Adding some noise
current_salary = base_salary + salary_noise

# Create the DataFrame
df = pd.DataFrame({
    "Age": age,
    "Years_of_Experience": years_of_experience,
    "Salary": current_salary
})

# Separate features and target column as per your example
features = ["Age", "Years_of_Experience"]
X = df[features]
y = df["Salary"]

# Display the Features and Target
print("Features:\n", X.head())  # Display first 5 rows of features
print("Target:\n", y.head())    # Display first 5 rows of target (Salary)

# Save the dataset to a CSV file
df.to_csv("salary_predictor_model.csv", index=False)
print("Synthetic dataset generated and saved as 'salary_predictor_model.csv'!")
