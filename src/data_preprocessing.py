"""
data_preprocessing.py
---------------------
Handles dataset loading, validation, missing value imputation,
feature separation, train-test splitting, and feature scaling.
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Standard feature set expected by the model
FEATURE_COLUMNS: List[str] = [
    "CGPA",
    "Tenth_Percentage",
    "Twelfth_Percentage",
    "Backlogs",
    "Python",
    "Cpp",
    "Java",
    "SQL",
    "Excel",
    "HTML_CSS",
    "JavaScript",
    "ML_Knowledge",
    "Data_Analysis",
    "Projects",
    "Internships",
    "Certifications",
    "Aptitude_Score",
    "Communication_Score"
]

TARGET_COLUMN: str = "Placement_Status"

NUMERICAL_COLS_FOR_SCALING: List[str] = [
    "CGPA",
    "Tenth_Percentage",
    "Twelfth_Percentage",
    "Aptitude_Score",
    "Communication_Score"
]

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file with error handling.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset at {csv_path} is empty.")
    return df

def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates whether required feature and target columns exist.
    """
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    is_valid = len(missing) == 0
    return is_valid, missing

def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing values and imputes them appropriately:
    Median for numerical columns, mode for binary/categorical.
    """
    cleaned_df = df.copy()
    
    # Check if there are missing values
    if cleaned_df[FEATURE_COLUMNS].isnull().values.any():
        for col in FEATURE_COLUMNS:
            if cleaned_df[col].isnull().sum() > 0:
                if col in NUMERICAL_COLS_FOR_SCALING:
                    median_val = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                else:
                    mode_val = cleaned_df[col].mode()[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                    
    if TARGET_COLUMN in cleaned_df.columns and cleaned_df[TARGET_COLUMN].isnull().sum() > 0:
        cleaned_df = cleaned_df.dropna(subset=[TARGET_COLUMN])
        
    return cleaned_df

def prepare_data_for_training(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Cleans data, separates X and y, splits into train/test sets,
    and applies standard scaling to numerical features.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    df_clean = clean_and_impute_data(df)
    
    X = df_clean[FEATURE_COLUMNS].copy()
    y = df_clean[TARGET_COLUMN].copy().astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Standardize numerical features for algorithms sensitive to scale (Logistic Regression)
    scaler = StandardScaler()
    
    # Scale train and test
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[NUMERICAL_COLS_FOR_SCALING] = scaler.fit_transform(X_train[NUMERICAL_COLS_FOR_SCALING])
    X_test_scaled[NUMERICAL_COLS_FOR_SCALING] = scaler.transform(X_test[NUMERICAL_COLS_FOR_SCALING])
    
    return (
        X_train_scaled.values,
        X_test_scaled.values,
        y_train.values,
        y_test.values,
        scaler,
        FEATURE_COLUMNS
    )

def transform_single_student_input(
    student_dict: Dict[str, Any],
    scaler: StandardScaler
) -> np.ndarray:
    """
    Transforms a single student profile dictionary into a scaled 2D array
    matching the trained model's feature order.
    """
    row = []
    for col in FEATURE_COLUMNS:
        val = student_dict.get(col, 0)
        row.append(float(val))
    
    student_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    student_df[NUMERICAL_COLS_FOR_SCALING] = scaler.transform(student_df[NUMERICAL_COLS_FOR_SCALING])
    return student_df.values
