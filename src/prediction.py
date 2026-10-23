"""
prediction.py
-------------
Provides inference services for placement possibility analysis.
Loads the trained model bundle and evaluates a candidate student profile.
"""

import os
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import transform_single_student_input, FEATURE_COLUMNS

_CACHED_BUNDLE: Optional[Dict[str, Any]] = None

def get_default_model_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, "models", "placement_model.pkl")

def load_trained_model(model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads and caches the model bundle from disk.
    """
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is not None and model_path is None:
        return _CACHED_BUNDLE
        
    path_to_use = model_path or get_default_model_path()
    if not os.path.exists(path_to_use):
        raise FileNotFoundError(
            f"Trained model bundle not found at: {path_to_use}. "
            "Please train the model first by running src/model_training.py."
        )
        
    bundle = joblib.load(path_to_use)
    if model_path is None:
        _CACHED_BUNDLE = bundle
    return bundle

def analyze_profile_factors(student_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Generates rule-based transparent insights explaining which factors
    favor the profile and which areas require improvement.
    """
    strengths = []
    concerns = []
    
    cgpa = float(student_data.get("CGPA", 0))
    backlogs = int(student_data.get("Backlogs", 0))
    projects = int(student_data.get("Projects", 0))
    internships = int(student_data.get("Internships", 0))
    certifications = int(student_data.get("Certifications", 0))
    aptitude = float(student_data.get("Aptitude_Score", 0))
    comm = float(student_data.get("Communication_Score", 0))
    
    tech_skills = ["Python", "Cpp", "Java", "SQL", "Excel", "HTML_CSS", "JavaScript", "ML_Knowledge", "Data_Analysis"]
    known_tech = [s for s in tech_skills if int(student_data.get(s, 0)) == 1]
    
    # Academic factors
    if cgpa >= 7.8:
        strengths.append(f"Strong academic foundation with CGPA {cgpa:.2f}")
    elif cgpa < 6.5:
        concerns.append(f"CGPA is {cgpa:.2f}; many campus recruitment drives specify a 6.5 or 7.0 minimum cutoff.")
        
    if backlogs == 0:
        strengths.append("Clean academic record with zero active backlogs.")
    else:
        concerns.append(f"{backlogs} active backlog(s) recorded; backlog clearance is mandatory for most corporate recruitment eligibility.")
        
    # Technical skills
    if len(known_tech) >= 4:
        strengths.append(f"Diverse technical toolset across {len(known_tech)} areas: {', '.join(known_tech[:4])}")
    elif len(known_tech) <= 1:
        concerns.append("Limited core technical proficiency detected. Expanding skills in Python, SQL, or Java is recommended.")
        
    # Practical experience
    if internships > 0:
        strengths.append(f"Valuable industrial exposure through {internships} internship(s).")
    if projects >= 2:
        strengths.append(f"Strong practical demonstration with {projects} hands-on technical project(s).")
    elif projects == 0:
        concerns.append("No technical projects listed. Capstone or course projects are critical during interview evaluations.")
        
    # Aptitude and Communication
    if aptitude >= 70:
        strengths.append(f"High aptitude test capability ({aptitude:.0f}/100) indicates strong first-round screening readiness.")
    elif aptitude < 50:
        concerns.append(f"Aptitude score ({aptitude:.0f}/100) is below standard online screening assessment cutoffs (~60%).")
        
    if comm >= 70:
        strengths.append("Solid communication rating, facilitating positive technical and HR interview rounds.")
    elif comm < 50:
        concerns.append("Communication score indicates interview practice and GD preparation will be advantageous.")
        
    return {
        "strengths": strengths,
        "concerns": concerns
    }

def predict_placement_possibility(
    student_data: Dict[str, Any],
    model_bundle: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Runs model inference on a student profile and returns comprehensive placement analysis.
    """
    bundle = model_bundle or load_trained_model()
    model = bundle["best_model"]
    scaler = bundle["scaler"]
    model_name = bundle["best_model_name"]
    
    # 1. Scale input features
    scaled_features = transform_single_student_input(student_data, scaler)
    
    # 2. Predict probability & label
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_features)[0]
        # Class 1 is "Placed" / "High Placement Possibility"
        prob_placement = float(probs[1]) if len(probs) > 1 else float(probs[0])
    else:
        # Fallback if model doesn't support predict_proba
        pred_raw = model.predict(scaled_features)[0]
        prob_placement = 0.85 if pred_raw == 1 else 0.25
        
    pred_label = 1 if prob_placement >= 0.50 else 0
    percentage = round(prob_placement * 100, 1)
    
    # 3. Categorize placement band
    if percentage >= 75.0:
        band = "High Placement Possibility"
        status_color = "green"
        summary_text = (
            "Based on the provided academic record, technical skills, and assessment scores, "
            "the model evaluates a relatively high placement possibility. "
            "Profile meets or exceeds standard benchmark screening parameters."
        )
    elif percentage >= 50.0:
        band = "Moderate Placement Possibility"
        status_color = "orange"
        summary_text = (
            "The model identifies a moderate placement possibility. "
            "While core criteria are satisfied, strengthening technical project depth "
            "or clearing active gaps will significantly enhance competitive advantage."
        )
    else:
        band = "Low Placement Possibility (Needs Skill & Academic Enhancement)"
        status_color = "red"
        summary_text = (
            "The model indicates that current profile parameters fall below typical corporate "
            "placement thresholds. Focused preparation in aptitude fundamentals, core technical "
            "skills, and project implementation is strongly advised."
        )
        
    # 4. Extract explanatory profile factors
    factors = analyze_profile_factors(student_data)
    
    return {
        "model_used": model_name,
        "placement_status_code": pred_label,
        "placement_possibility_percentage": percentage,
        "placement_band": band,
        "status_color": status_color,
        "summary": summary_text,
        "strengths": factors["strengths"],
        "improvement_areas": factors["concerns"],
        "disclaimer": (
            "Note: This prediction is a statistical estimate produced by machine learning "
            "trained on benchmark placement patterns. It is an advisory decision-support tool "
            "and does not constitute a legal or contractual employment guarantee."
        )
    }
