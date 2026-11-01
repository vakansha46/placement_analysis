"""
model_training.py
-----------------
Trains and compares multiple machine learning classification algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Calculates real metrics (Accuracy, Precision, Recall, F1, Confusion Matrix),
identifies the best model, extracts feature importances, and persists the
trained artifacts using joblib.
"""

import os
import sys
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data_preprocessing import (
    load_dataset,
    prepare_data_for_training,
    FEATURE_COLUMNS
)

def train_and_evaluate_models(csv_path: str, output_model_dir: str) -> Dict[str, Any]:
    """
    Loads data, trains Logistic Regression, Decision Tree, and Random Forest,
    evaluates them on the test split, selects the best model, and saves all artifacts.
    """
    os.makedirs(output_model_dir, exist_ok=True)
    
    # 1. Load and prepare data
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data_for_training(df)
    
    # 2. Define the candidate models
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42
        )
    }
    
    evaluation_results: Dict[str, Dict[str, Any]] = {}
    fitted_models: Dict[str, Any] = {}
    
    # 3. Train and evaluate each model on real data
    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        evaluation_results[model_name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "Confusion_Matrix": cm
        }
        fitted_models[model_name] = model
        
    # 4. Model Selection: Pick the highest F1-score (with Accuracy tie-breaker)
    best_model_name = max(
        evaluation_results.keys(),
        key=lambda m: (evaluation_results[m]["F1_Score"], evaluation_results[m]["Accuracy"])
    )
    best_model = fitted_models[best_model_name]
    
    # 5. Extract Feature Importances / Coefficients for Explainability
    feature_importance_dict: Dict[str, float] = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        for feat, imp in zip(feature_names, importances):
            feature_importance_dict[feat] = round(float(imp), 4)
    elif hasattr(best_model, "coef_"):
        coeffs = np.abs(best_model.coef_[0])
        total_c = np.sum(coeffs) if np.sum(coeffs) > 0 else 1.0
        normalized_coeffs = coeffs / total_c
        for feat, coeff in zip(feature_names, normalized_coeffs):
            feature_importance_dict[feat] = round(float(coeff), 4)
            
    sorted_features = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

    # 6. Save Bundle and Metrics
    model_bundle = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "fitted_models": fitted_models,
        "scaler": scaler,
        "feature_names": feature_names,
        "evaluation_results": evaluation_results,
        "feature_importances": sorted_features,
        "train_samples_count": len(X_train),
        "test_samples_count": len(X_test)
    }
    
    bundle_path = os.path.join(output_model_dir, "placement_model.pkl")
    joblib.dump(model_bundle, bundle_path)
    
    metrics_path = os.path.join(output_model_dir, "model_metrics.json")
    json_summary = {
        "best_model_name": best_model_name,
        "evaluation_results": evaluation_results,
        "top_feature_importances": sorted_features,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4)
        
    print(f"=== Model Training Summary ===")
    print(f"Best Selected Model: {best_model_name}")
    for name, m in evaluation_results.items():
        print(f"[{name}] Acc: {m['Accuracy']:.2%}, Prec: {m['Precision']:.2%}, Rec: {m['Recall']:.2%}, F1: {m['F1_Score']:.2%}")
    print(f"Artifacts saved to:\n  - {bundle_path}\n  - {metrics_path}")
    
    return model_bundle

if __name__ == "__main__":
    data_file = os.path.join(project_root, "data", "students.csv")
    models_dir = os.path.join(project_root, "models")
    
    train_and_evaluate_models(data_file, models_dir)
