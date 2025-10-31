import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Global variables to store column names (initialized here)
CATEGORICAL_COLS = []
FEATURE_COLS = []
RANDOM_STATE = 42

# --- Preprocessing Maps (Added for Ordinal Encoding) ---
EDUCATION_MAP = {
    "Bachelor's": "Bachelor's Degree",
    "Master's": "Master's Degree",
    "PhD": "PhD",
    "phD": "PhD",
    "Bachelor's Degree": "Bachelor's Degree",
    "Master's Degree": "Master's Degree",
    "High School": "High School"
}

ORDINAL_ORDER = {
    'High School': 0,
    "Bachelor's Degree": 1,
    "Master's Degree": 2,
    "PhD": 3
}
# -------------------------------------------------------

# --- Phase 1: Data Loading ---
def load_data(file_name='Salary_Data.csv'):
    """Loads the dataset and handles potential delimiter issues."""
    try:
        # Assuming the file is accessible in the current context
        df = pd.read_csv(file_name)
        
        # Check for common delimiter issue (single column)
        if df.shape[1] <= 2 and 'Salary' not in df.columns: 
             df = pd.read_csv(file_name, sep=';')
             
        print(f"[STATUS] Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] {file_name} not found. Please ensure the file is correctly uploaded or placed.")
        return None

# --- Phase 2: Data Cleaning and Feature Engineering ---
def prepare_data(df):
    """Cleans the data and performs Feature Engineering."""
    print("\n--- Phase 2: Data Cleaning and Feature Engineering ---")
    
    # Drop rows with missing values (since the count was very low)
    df_cleaned = df.dropna().copy()
    print(f"Data cleaned. Remaining rows: {df_cleaned.shape[0]}")
    
    # 1. Standardize and Ordinal Encode Education Level
    df_cleaned['Education Level'] = df_cleaned['Education Level'].replace(EDUCATION_MAP)
    df_cleaned['Education_Encoded'] = df_cleaned['Education Level'].map(ORDINAL_ORDER)
    
    # 2. Define categorical columns for One-Hot Encoding
    # Only Gender and Job Title are left as nominal categorical features
    global CATEGORICAL_COLS
    CATEGORICAL_COLS = ['Gender', 'Job Title']

    # 3. Apply One-Hot Encoding to the nominal categorical columns
    df_encoded = pd.get_dummies(df_cleaned, columns=CATEGORICAL_COLS, drop_first=True)
    
    # 4. Drop the original 'Education Level' column (now redundant)
    df_final = df_encoded.drop(columns=['Education Level'])
    
    # Store the final column names globally for alignment in predictions (CRITICAL)
    global FEATURE_COLS
    FEATURE_COLS = df_final.drop('Salary', axis=1).columns

    # The resulting features now include: Age, Years of Experience, Education_Encoded, Gender_dummies, Job_Title_dummies
    return df_final

# --- Phase 3: Train the BEST Model (Random Forest) and Evaluate ---
def train_best_model(df_encoded):
    """Trains the Random Forest Regressor model and evaluates it using train-test split."""
    print("\n--- Phase 3: Training Best Model (Random Forest Regressor) ---")
    
    X = df_encoded.drop('Salary', axis=1)
    y = df_encoded['Salary']
    
    # 1. Perform Train-Test Split (CRITICAL for valid evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 2. Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1) 
    model.fit(X_train, y_train)
    
    # 3. Predict on the Test Set
    y_pred = model.predict(X_test)
    
    # 4. Calculate and Report Metrics (MAE, MSE, RMSE, R2)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation (on Test Set) ---")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Mean Squared Error (MSE): ${mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print("-------------------------------------")
    
    return model

# --- Phase 4: Predict New Data ---
def predict_new_salary(model, new_data):
    """
    Predicts salary for a single new employee using the trained Random Forest model.
    """
    print("\n\n--- Phase 4: Predicting New Salary ---")
    
    # 1. Convert the new data point into a DataFrame
    new_df = pd.DataFrame([new_data])
    
    # 1a. Apply Education Standardization and Ordinal Encoding (New Step)
    new_df['Education Level'] = new_df['Education Level'].replace(EDUCATION_MAP)
    new_df['Education_Encoded'] = new_df['Education Level'].map(ORDINAL_ORDER)
    new_df = new_df.drop(columns=['Education Level']) # Drop original column
    
    # 2. Replicate One-Hot Encoding on the nominal features (Gender, Job Title)
    # Note: drop_first=True must match training
    new_df_encoded = pd.get_dummies(new_df, columns=CATEGORICAL_COLS, drop_first=True)
    
    # 3. Align the columns with the training data columns (CRITICAL STEP)
    # Create an empty DataFrame with all training feature columns, initialized to 0
    X_new = pd.DataFrame(0, index=new_df_encoded.index, columns=FEATURE_COLS)
    
    # Transfer the encoded features (where they exist) to the aligned DataFrame
    for col in new_df_encoded.columns:
        if col in X_new.columns:
            X_new[col] = new_df_encoded[col]

    # 4. Make Prediction
    predicted_salary = model.predict(X_new)[0]
    
    # 5. Print Result
    print(f"Prediction for Employee:")
    for key, value in new_data.items():
        print(f"  {key}: {value}")
        
    print("-" * 30)
    print(f"Predicted Salary: ${predicted_salary:,.2f}")
    print("-" * 30)


# --- New Function for User Input ---
def get_user_input():
    """Collects feature data from the user with basic validation."""
    print("\n--- Enter Employee Data for Prediction ---")
    
    # Helper to get valid float input (for Age and Experience)
    def get_float_input(prompt):
        while True:
            try:
                value = float(input(prompt))
                return value
            except ValueError:
                print("Invalid input. Please enter a number (e.g., 5.0, 35.5).")

    # Helper to get valid string input
    def get_string_input(prompt):
        return input(prompt).strip()

    age = get_float_input("Enter Age (e.g., 35.0): ")
    exp = get_float_input("Enter Years of Experience (e.g., 10.0): ")
    gender = get_string_input("Enter Gender (Male/Female/Other): ")
    education = get_string_input("Enter Education Level (e.g., Bachelor's Degree, Master's Degree, PhD): ")
    job_title = get_string_input("Enter Job Title (e.g., Data Scientist, Software Engineer): ")
    
    # Create dictionary matching the structure expected by the prediction function
    user_data = {
        'Age': age,
        'Years of Experience': exp,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title
    }
    return user_data


# --- Main Execution Block ---
if __name__ == "__main__":
    
    salary_df = load_data() 
    
    if salary_df is not None:
        processed_df = prepare_data(salary_df)
        
        # Train and evaluate the model
        best_model = train_best_model(processed_df)
        
        # Interactive loop for user predictions
        while True:
            user_employee_data = get_user_input()
            
            # Check for valid Education Level input before prediction
            edu_level = user_employee_data.get('Education Level')
            # Check for both standardized and original forms
            if edu_level not in ORDINAL_ORDER and EDUCATION_MAP.get(edu_level) not in ORDINAL_ORDER:
                 print(f"[WARNING] Input Education Level '{edu_level}' not recognized by the model. This may lead to an inaccurate prediction.")

            predict_new_salary(best_model, user_employee_data)
            
            cont = input("\nPredict another salary? (y/n): ")
            if cont.lower() != 'y':
                break
