"""
skill_matching.py
-----------------
Performs job role and company requirement matching against a student's profile.
Implements set intersection and set difference logic for transparent skill gap calculation:
- Matching Skills = Student Skills ∩ Required Skills
- Skill Gap = Required Skills − Student Skills
- Match % = (len(Matching Skills) / len(Required Skills)) * 100
"""

import os
from typing import List, Dict, Any, Set, Tuple
import pandas as pd

def get_default_job_roles_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, "data", "job_roles.csv")

def load_job_roles(csv_path: str = None) -> pd.DataFrame:
    """
    Loads job roles dataset from CSV.
    """
    path = csv_path or get_default_job_roles_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Job roles dataset not found at: {path}")
    df = pd.read_csv(path)
    return df

def parse_skill_list(skills_str: str) -> List[str]:
    """
    Splits comma-separated string of skills into a clean, normalized list.
    """
    if pd.isna(skills_str):
        return []
    return [s.strip() for s in str(skills_str).split(",") if s.strip()]

def calculate_role_match(
    student_skills: List[str],
    student_cgpa: float,
    role_row: pd.Series
) -> Dict[str, Any]:
    """
    Calculates detailed skill intersection, difference, and percentage match
    for a single job role against student profile.
    """
    required_skills = parse_skill_list(role_row["Required_Skills"])
    
    # Case-insensitive set comparison
    student_set_lower = {s.lower(): s for s in student_skills}
    required_set_lower = {s.lower(): s for s in required_skills}
    
    matching_lower = set(student_set_lower.keys()).intersection(set(required_set_lower.keys()))
    missing_lower = set(required_set_lower.keys()) - set(student_set_lower.keys())
    
    # Retrieve original casing
    matching_skills = [required_set_lower[k] for k in sorted(matching_lower)]
    missing_skills = [required_set_lower[k] for k in sorted(missing_lower)]
    
    total_req = len(required_skills)
    match_pct = round((len(matching_skills) / total_req * 100.0), 1) if total_req > 0 else 0.0
    
    min_cgpa = float(role_row.get("Min_CGPA", 6.0))
    is_cgpa_eligible = student_cgpa >= min_cgpa
    
    # Sample companies
    companies = role_row.get("Sample_Companies", "General Industry")
    category = role_row.get("Category", "General")
    
    return {
        "role_id": role_row.get("Role_ID", ""),
        "role_name": role_row["Role_Name"],
        "category": category,
        "match_percentage": match_pct,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "total_required_skills": total_req,
        "min_cgpa": min_cgpa,
        "is_cgpa_eligible": is_cgpa_eligible,
        "sample_companies": companies,
        "description": role_row.get("Description", "")
    }

def match_all_job_roles(
    student_skills: List[str],
    student_cgpa: float,
    job_roles_df: pd.DataFrame = None
) -> List[Dict[str, Any]]:
    """
    Evaluates all predefined job roles against student profile
    and returns a list sorted by match percentage descending.
    """
    df = job_roles_df if job_roles_df is not None else load_job_roles()
    results = []
    
    for _, row in df.iterrows():
        match_info = calculate_role_match(student_skills, student_cgpa, row)
        results.append(match_info)
        
    # Sort primarily by match_percentage descending, secondarily by CGPA eligibility
    results.sort(key=lambda x: (x["match_percentage"], 1 if x["is_cgpa_eligible"] else 0), reverse=True)
    return results

def get_role_details_by_name(
    role_name: str,
    job_roles_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Fetches raw role information by name.
    """
    df = job_roles_df if job_roles_df is not None else load_job_roles()
    match = df[df["Role_Name"].str.lower() == role_name.strip().lower()]
    if match.empty:
        raise ValueError(f"Job role '{role_name}' not found in predefined database.")
    row = match.iloc[0]
    return {
        "role_id": row.get("Role_ID", ""),
        "role_name": row["Role_Name"],
        "category": row.get("Category", ""),
        "required_skills": parse_skill_list(row["Required_Skills"]),
        "min_cgpa": float(row.get("Min_CGPA", 6.0)),
        "sample_companies": row.get("Sample_Companies", ""),
        "description": row.get("Description", "")
    }
