"""
generate_dataset.py
-------------------
Generates a realistic synthetic dataset of student academic, technical,
and professional profiles for the Placement Analysis and Skill Gap System.

NOTE: This dataset contains synthetic benchmark data generated for academic
demonstration and viva explanation purposes. It is clearly documented as synthetic
and does NOT claim to represent private institutional records.
"""

import os
import numpy as np
import pandas as pd

def generate_student_dataset(n_samples: int = 700, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    # 1. Academic records
    # CGPA follows a realistic distribution centered around 7.2
    cgpa = np.clip(np.random.normal(loc=7.1, scale=1.1, size=n_samples), 5.0, 9.9).round(2)

    # 10th and 12th percentages correlate with academic capabilities
    tenth_pct = np.clip(cgpa * 8.0 + np.random.normal(15, 6, size=n_samples), 50.0, 98.0).round(1)
    twelfth_pct = np.clip(cgpa * 7.8 + np.random.normal(16, 7, size=n_samples), 50.0, 97.0).round(1)

    # Backlogs: lower CGPA students have higher likelihood of backlogs
    backlogs = []
    p_low = [0.35, 0.30, 0.18, 0.11, 0.06]
    p_high = [0.85, 0.10, 0.03, 0.015, 0.005]
    for c in cgpa:
        p = p_low if c < 6.8 else p_high
        backlogs.append(np.random.choice([0, 1, 2, 3, 4], p=p))
    backlogs = np.array(backlogs)

    # 2. Technical Skills (Binary 1=Proficient, 0=Not Proficient)
    skill_bias = (cgpa - 5.0) / 4.9  # 0 to 1

    p_python = np.clip(0.35 + skill_bias * 0.40, 0.15, 0.90)
    p_cpp = np.clip(0.30 + skill_bias * 0.35, 0.15, 0.85)
    p_java = np.clip(0.28 + skill_bias * 0.35, 0.15, 0.85)
    p_sql = np.clip(0.30 + skill_bias * 0.40, 0.15, 0.90)
    p_excel = np.clip(0.40 + skill_bias * 0.35, 0.25, 0.90)
    p_html_css = np.clip(0.40 + skill_bias * 0.35, 0.20, 0.90)
    p_js = np.clip(0.28 + skill_bias * 0.35, 0.15, 0.85)
    p_ml = np.clip(0.12 + skill_bias * 0.45, 0.05, 0.80)
    p_data_analysis = np.clip(0.18 + skill_bias * 0.45, 0.08, 0.85)

    python_skill = (np.random.rand(n_samples) < p_python).astype(int)
    cpp_skill = (np.random.rand(n_samples) < p_cpp).astype(int)
    java_skill = (np.random.rand(n_samples) < p_java).astype(int)
    sql_skill = (np.random.rand(n_samples) < p_sql).astype(int)
    excel_skill = (np.random.rand(n_samples) < p_excel).astype(int)
    html_css_skill = (np.random.rand(n_samples) < p_html_css).astype(int)
    js_skill = (np.random.rand(n_samples) < p_js).astype(int)
    ml_skill = (np.random.rand(n_samples) < p_ml).astype(int)
    data_analysis_skill = (np.random.rand(n_samples) < p_data_analysis).astype(int)

    # 3. Practical Profile Information
    projects = np.clip(np.random.poisson(lam=1.5 + skill_bias * 1.2), 0, 5)
    internships = np.random.choice([0, 1, 2, 3], p=[0.60, 0.26, 0.11, 0.03], size=n_samples)
    certifications = np.clip(np.random.poisson(lam=1.0 + skill_bias * 1.0), 0, 5)

    aptitude_score = np.clip(np.random.normal(loc=50 + cgpa * 3.2, scale=13, size=n_samples), 30, 98).round(1)
    communication_score = np.clip(np.random.normal(loc=60, scale=15, size=n_samples), 30, 98).round(1)

    # 4. Realistic Placement Likelihood Calculation
    tech_skills_sum = (python_skill + cpp_skill + java_skill + sql_skill +
                       excel_skill + html_css_skill + js_skill + ml_skill + data_analysis_skill)

    academic_factor = (cgpa - 6.5) * 1.3
    backlog_penalty = backlogs * 1.8
    tech_factor = (tech_skills_sum - 4.0) * 0.45
    exp_factor = (projects * 0.4 + internships * 1.1 + certifications * 0.25)
    test_factor = ((aptitude_score - 65) / 15.0 * 0.8) + ((communication_score - 65) / 15.0 * 0.7)

    composite_latent = (academic_factor + tech_factor + exp_factor + test_factor - backlog_penalty - 0.2)
    
    # Sigmoid with noise
    noise = np.random.normal(0, 0.75, size=n_samples)
    probability = 1.0 / (1.0 + np.exp(-(composite_latent + noise)))

    placement_status = (probability >= 0.50).astype(int)

    first_names = [
        "Aarav", "Aditi", "Ananya", "Arjun", "Deepak", "Divya", "Ishaan", "Kavya",
        "Manish", "Meera", "Neha", "Nikhil", "Pooja", "Pranav", "Rahul", "Rhea",
        "Rohan", "Sanjana", "Siddharth", "Sneha", "Tanvi", "Varun", "Vikas", "Yash"
    ]
    student_ids = [f"STU{1001 + i}" for i in range(n_samples)]
    student_names = [f"{np.random.choice(first_names)} {chr(65 + (i % 26))}." for i in range(n_samples)]

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Student_Name": student_names,
        "CGPA": cgpa,
        "Tenth_Percentage": tenth_pct,
        "Twelfth_Percentage": twelfth_pct,
        "Backlogs": backlogs,
        "Python": python_skill,
        "Cpp": cpp_skill,
        "Java": java_skill,
        "SQL": sql_skill,
        "Excel": excel_skill,
        "HTML_CSS": html_css_skill,
        "JavaScript": js_skill,
        "ML_Knowledge": ml_skill,
        "Data_Analysis": data_analysis_skill,
        "Projects": projects,
        "Internships": internships,
        "Certifications": certifications,
        "Aptitude_Score": aptitude_score,
        "Communication_Score": communication_score,
        "Placement_Status": placement_status
    })

    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "students.csv")

    df = generate_student_dataset(n_samples=700, random_state=42)
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {len(df)} student records at: {csv_path}")
    print("Class distribution:")
    print(df["Placement_Status"].value_counts(normalize=True))
    print(df["Placement_Status"].value_counts())
