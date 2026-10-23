"""
app.py
------
Student Placement Analysis and Skill Gap Recommendation System
Built with Streamlit, Scikit-learn, Pandas, and NumPy.

A clean, explainable, and realistic decision-support dashboard for college students.
"""

import os
import sys
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src modules are resolvable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data_preprocessing import load_dataset, FEATURE_COLUMNS
from src.prediction import load_trained_model, predict_placement_possibility
from src.skill_matching import load_job_roles, match_all_job_roles, calculate_role_match
from src.recommendations import get_recommendations_for_skills, get_enrichment_recommendations
from src.model_training import train_and_evaluate_models

# Streamlit Page Configuration
st.set_page_config(
    page_title="Student Placement Analysis & Skill Gap System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-danger {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 15px;
        font-size: 0.92rem;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get paths
def get_paths():
    data_dir = os.path.join(current_dir, "data")
    models_dir = os.path.join(current_dir, "models")
    return {
        "students_csv": os.path.join(data_dir, "students.csv"),
        "job_roles_csv": os.path.join(data_dir, "job_roles.csv"),
        "resources_csv": os.path.join(data_dir, "learning_resources.csv"),
        "model_bundle": os.path.join(models_dir, "placement_model.pkl"),
        "models_dir": models_dir
    }

# Safe model loader with Streamlit caching
@st.cache_resource
def get_cached_model():
    paths = get_paths()
    if not os.path.exists(paths["model_bundle"]):
        # Auto-train if model bundle not yet present
        if os.path.exists(paths["students_csv"]):
            return train_and_evaluate_models(paths["students_csv"], paths["models_dir"])
        else:
            return None
    return load_trained_model(paths["model_bundle"])

# Initialize default session state
if "student_profile" not in st.session_state:
    st.session_state.student_profile = {
        "Student_Name": "Rohan S.",
        "CGPA": 7.8,
        "Tenth_Percentage": 82.0,
        "Twelfth_Percentage": 79.5,
        "Backlogs": 0,
        "Python": 1,
        "Cpp": 0,
        "Java": 0,
        "SQL": 1,
        "Excel": 1,
        "HTML_CSS": 0,
        "JavaScript": 0,
        "ML_Knowledge": 0,
        "Data_Analysis": 1,
        "Projects": 2,
        "Internships": 1,
        "Certifications": 1,
        "Aptitude_Score": 75.0,
        "Communication_Score": 72.0,
        "Target_Role": "Data Analyst"
    }

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=70)
st.sidebar.title("Placement System")
st.sidebar.markdown("**Decision-Support & Skill Gap Platform**")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home & Project Overview",
        "📝 Student Profile & Placement Analysis",
        "💼 Job Role & Company Matching",
        "🎯 Skill Gap Analysis",
        "📚 Learning Recommendations",
        "📊 Model Performance & Evaluation",
        "ℹ️ About Project & Viva Guide"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("🎓 B.Tech CSE Final Year Project")
st.sidebar.caption("Python • Scikit-learn • Streamlit")

paths = get_paths()
model_bundle = get_cached_model()

# ==============================================================================
# 1. HOME & PROJECT OVERVIEW
# ==============================================================================
if nav_choice == "🏠 Home & Project Overview":
    st.markdown('<div class="main-title">Student Placement Analysis & Skill Gap Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A transparent, machine-learning-assisted decision support system for undergraduate engineering students.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin:0; color:#4B5563;">Benchmark Cohort</h4>
            <h2 style="margin:8px 0; color:#1E3A8A;">700 Students</h2>
            <p style="margin:0; font-size:0.85rem; color:#6B7280;">Synthetic training benchmark</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin:0; color:#4B5563;">ML Algorithms</h4>
            <h2 style="margin:8px 0; color:#1E3A8A;">3 Evaluated</h2>
            <p style="margin:0; font-size:0.85rem; color:#6B7280;">Logistic Reg, Decision Tree, RF</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin:0; color:#4B5563;">Predefined Roles</h4>
            <h2 style="margin:8px 0; color:#1E3A8A;">8 Industry Roles</h2>
            <p style="margin:0; font-size:0.85rem; color:#6B7280;">Data, Software, Web, Cloud</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        best_name = model_bundle["best_model_name"] if model_bundle else "Random Forest"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#4B5563;">Best Model</h4>
            <h2 style="margin:8px 0; color:#1E3A8A;">{best_name}</h2>
            <p style="margin:0; font-size:0.85rem; color:#6B7280;">Selected via test F1-score</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📌 Motivation & Research Gap")
    st.write(
        "Many existing placement analysis systems stop after outputting a binary 'Placed / Not Placed' prediction. "
        "This leaves the student without actionable guidance on **why** that outcome was predicted, "
        "**what job roles** suit their current profile, and **which missing skills** need to be improved. "
        "Our project addresses this research gap by integrating **Placement Analysis**, **Job Role Matching**, "
        "**Set-Theoretic Skill Gap Identification**, and **Curated Learning Recommendations** into one cohesive platform."
    )

    st.markdown("### 🔄 Complete Workflow Architecture")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**Step 1: Student Profile**\n\nEnter academic grades (CGPA, 10th, 12th), backlogs, technical skills, projects, and assessment scores.")
    with c2:
        st.info("**Step 2: ML Placement Analysis**\n\nStandardized feature scaling + trained classification model calculates placement likelihood and factor breakdown.")
    with c3:
        st.info("**Step 3: Role & Skill Gap Analysis**\n\nPerforms set intersection & set difference against predefined job roles to determine matching and missing skills.")
    with c4:
        st.info("**Step 4: Learning Recommendations**\n\nMaps skill gaps to curated educational tutorials, official documentation, and practical coding milestones.")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Academic Notice:</strong> This system is designed as an advisory decision-support platform.
        All predictions and match percentages are statistical estimates derived from training patterns.
        The system does <strong>not</strong> promise or guarantee employment.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. STUDENT PROFILE & PLACEMENT ANALYSIS
# ==============================================================================
elif nav_choice == "📝 Student Profile & Placement Analysis":
    st.markdown('<div class="main-title">Student Profile & Placement Possibility Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Provide student academic details, skills, and assessment scores for machine-learning analysis.</div>', unsafe_allow_html=True)

    curr = st.session_state.student_profile

    with st.form("student_profile_form"):
        st.subheader("1. Academic Information")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            name = st.text_input("Student Name", value=curr.get("Student_Name", "Student Candidate"))
        with a2:
            cgpa = st.number_input("B.Tech CGPA (0.0 to 10.0)", min_value=0.0, max_value=10.0, value=float(curr.get("CGPA", 7.5)), step=0.1)
        with a3:
            tenth = st.number_input("10th Standard Percentage", min_value=40.0, max_value=100.0, value=float(curr.get("Tenth_Percentage", 80.0)), step=0.5)
        with a4:
            twelfth = st.number_input("12th / Diploma Percentage", min_value=40.0, max_value=100.0, value=float(curr.get("Twelfth_Percentage", 78.0)), step=0.5)

        backlogs = st.slider("Number of Active Backlogs", min_value=0, max_value=6, value=int(curr.get("Backlogs", 0)))

        st.subheader("2. Technical & Programming Skills")
        st.caption("Select the programming languages and technical domains you are proficient in:")
        t1, t2, t3 = st.columns(3)
        with t1:
            has_python = st.checkbox("Python Programming", value=bool(curr.get("Python", 1)))
            has_cpp = st.checkbox("C / C++", value=bool(curr.get("Cpp", 0)))
            has_java = st.checkbox("Java", value=bool(curr.get("Java", 0)))
        with t2:
            has_sql = st.checkbox("SQL & Relational Databases", value=bool(curr.get("SQL", 1)))
            has_excel = st.checkbox("Microsoft Excel (Advanced / Formulas)", value=bool(curr.get("Excel", 1)))
            has_html_css = st.checkbox("HTML & CSS", value=bool(curr.get("HTML_CSS", 0)))
        with t3:
            has_js = st.checkbox("JavaScript", value=bool(curr.get("JavaScript", 0)))
            has_ml = st.checkbox("Machine Learning Knowledge", value=bool(curr.get("ML_Knowledge", 0)))
            has_data_analysis = st.checkbox("Data Analysis (Pandas/Statistics)", value=bool(curr.get("Data_Analysis", 1)))

        st.subheader("3. Practical Experience & Assessment Scores")
        p1, p2, p3 = st.columns(3)
        with p1:
            projects = st.number_input("Completed Technical Projects", min_value=0, max_value=10, value=int(curr.get("Projects", 2)))
        with p2:
            internships = st.number_input("Completed Internships", min_value=0, max_value=5, value=int(curr.get("Internships", 1)))
        with p3:
            certifications = st.number_input("Technical Certifications", min_value=0, max_value=10, value=int(curr.get("Certifications", 1)))

        s1, s2, s3 = st.columns(3)
        with s1:
            aptitude = st.slider("Aptitude / Quantitative Test Score (out of 100)", min_value=30.0, max_value=100.0, value=float(curr.get("Aptitude_Score", 75.0)), step=1.0)
        with s2:
            communication = st.slider("Communication / Soft Skills Score (out of 100)", min_value=30.0, max_value=100.0, value=float(curr.get("Communication_Score", 70.0)), step=1.0)
        with s3:
            job_roles_df = load_job_roles(paths["job_roles_csv"])
            all_role_names = job_roles_df["Role_Name"].tolist()
            default_role_idx = all_role_names.index(curr.get("Target_Role", "Data Analyst")) if curr.get("Target_Role") in all_role_names else 0
            target_role = st.selectbox("Preferred / Target Job Role", options=all_role_names, index=default_role_idx)

        submitted = st.form_submit_button("🔍 Analyze Student Profile")

    if submitted:
        st.session_state.student_profile = {
            "Student_Name": name,
            "CGPA": cgpa,
            "Tenth_Percentage": tenth,
            "Twelfth_Percentage": twelfth,
            "Backlogs": backlogs,
            "Python": int(has_python),
            "Cpp": int(has_cpp),
            "Java": int(has_java),
            "SQL": int(has_sql),
            "Excel": int(has_excel),
            "HTML_CSS": int(has_html_css),
            "JavaScript": int(has_js),
            "ML_Knowledge": int(has_ml),
            "Data_Analysis": int(has_data_analysis),
            "Projects": int(projects),
            "Internships": int(internships),
            "Certifications": int(certifications),
            "Aptitude_Score": aptitude,
            "Communication_Score": communication,
            "Target_Role": target_role
        }
        st.success("✅ Profile updated successfully! Showing Machine Learning Placement Analysis below.")

    # Run Prediction Analysis
    if model_bundle is not None:
        student_input = st.session_state.student_profile
        analysis = predict_placement_possibility(student_input, model_bundle)

        st.markdown("---")
        st.markdown("### 📊 Placement Possibility Outcome")

        r1, r2, r3 = st.columns([1.2, 2, 1.5])
        with r1:
            pct = analysis["placement_possibility_percentage"]
            st.metric(label="Placement Possibility", value=f"{pct}%")
            if pct >= 75.0:
                st.markdown('<span class="badge-success">High Placement Possibility</span>', unsafe_allow_html=True)
            elif pct >= 50.0:
                st.markdown('<span class="badge-warning">Moderate Placement Possibility</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-danger">Low Placement Possibility</span>', unsafe_allow_html=True)

        with r2:
            st.markdown(f"**Model Assessment ({analysis['model_used']}):**")
            st.write(analysis["summary"])
            st.progress(pct / 100.0)

        with r3:
            st.markdown("**Profile Quick Summary:**")
            st.write(f"- **CGPA:** {student_input['CGPA']} | **Backlogs:** {student_input['Backlogs']}")
            st.write(f"- **Projects:** {student_input['Projects']} | **Internships:** {student_input['Internships']}")
            st.write(f"- **Aptitude:** {student_input['Aptitude_Score']}/100")

        st.markdown("#### 🔍 Explainable Profile Factors")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("**✅ Positive Profile Strengths:**")
            if analysis["strengths"]:
                for s in analysis["strengths"]:
                    st.write(f"- {s}")
            else:
                st.write("- Academic profile meets baseline standards.")

        with f_col2:
            st.markdown("**⚠️ Recommended Improvement Areas:**")
            if analysis["improvement_areas"]:
                for c in analysis["improvement_areas"]:
                    st.write(f"- {c}")
            else:
                st.write("- No critical profile flags detected.")

        st.markdown(f'<div class="disclaimer-box">{analysis["disclaimer"]}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model not yet loaded. Please ensure dataset exists and model is trained.")

# ==============================================================================
# 3. JOB ROLE & COMPANY MATCHING
# ==============================================================================
elif nav_choice == "💼 Job Role & Company Matching":
    st.markdown('<div class="main-title">Job Role & Company Matching</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Compares student technical profile with predefined campus recruitment role benchmarks.</div>', unsafe_allow_html=True)

    curr = st.session_state.student_profile
    job_roles_df = load_job_roles(paths["job_roles_csv"])

    # Extract student skill list
    skill_mapping = {
        "Python": curr.get("Python", 0),
        "C++": curr.get("Cpp", 0),
        "Java": curr.get("Java", 0),
        "SQL": curr.get("SQL", 0),
        "Excel": curr.get("Excel", 0),
        "HTML": curr.get("HTML_CSS", 0),
        "CSS": curr.get("HTML_CSS", 0),
        "JavaScript": curr.get("JavaScript", 0),
        "Machine Learning": curr.get("ML_Knowledge", 0),
        "Data Analysis": curr.get("Data_Analysis", 0),
        "Pandas": curr.get("Data_Analysis", 0),
        "NumPy": curr.get("Python", 0) and curr.get("Data_Analysis", 0),
        "Statistics": curr.get("Data_Analysis", 0),
        "OOP": curr.get("Java", 0) or curr.get("Cpp", 0) or curr.get("Python", 0),
        "Git": curr.get("Projects", 0) >= 2,
        "Data Structures": curr.get("Cpp", 0) or curr.get("Java", 0)
    }
    student_skills = [k for k, v in skill_mapping.items() if bool(v)]

    st.markdown(f"**Identified Student Skills ({len(student_skills)}):** " + ", ".join(f"`{s}`" for s in student_skills))
    st.markdown(f"**Current Student CGPA:** `{curr.get('CGPA', 7.5)}`")

    # Match against all roles
    matches = match_all_job_roles(student_skills, float(curr.get("CGPA", 7.5)), job_roles_df)

    # Convert to presentation DataFrame
    table_data = []
    for m in matches:
        elig_badge = "✅ Eligible" if m["is_cgpa_eligible"] else f"⚠️ Below Min CGPA ({m['min_cgpa']})"
        table_data.append({
            "Job Role": m["role_name"],
            "Category": m["category"],
            "Profile Match": f"{m['match_percentage']}%",
            "Matching Skills Count": f"{len(m['matching_skills'])} / {m['total_required_skills']}",
            "Missing Skills Count": len(m["missing_skills"]),
            "Academic Eligibility": elig_badge,
            "Sample Hiring Companies": m["sample_companies"]
        })
    df_results = pd.DataFrame(table_data)

    st.dataframe(df_results, use_container_width=True)

    # Visualization of Match Percentages
    st.markdown("### 📊 Role Compatibility Distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    role_names = [m["role_name"] for m in matches]
    match_pcts = [m["match_percentage"] for m in matches]
    colors = ["#2563EB" if p >= 70 else "#F59E0B" if p >= 40 else "#EF4444" for p in match_pcts]

    bars = ax.barh(role_names[::-1], match_pcts[::-1], color=colors[::-1])
    ax.set_xlabel("Profile Match Percentage (%)")
    ax.set_xlim(0, 100)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Note on Predefined Companies:</strong> Company listings (e.g. ABC Technologies, TechNova)
        are provided as representative samples for academic demonstration and do not reflect active live vacancies.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. SKILL GAP ANALYSIS
# ==============================================================================
elif nav_choice == "🎯 Skill Gap Analysis":
    st.markdown('<div class="main-title">Skill Gap Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Identifies the exact difference between your current skills and target role requirements.</div>', unsafe_allow_html=True)

    curr = st.session_state.student_profile
    job_roles_df = load_job_roles(paths["job_roles_csv"])

    skill_mapping = {
        "Python": curr.get("Python", 0),
        "C++": curr.get("Cpp", 0),
        "Java": curr.get("Java", 0),
        "SQL": curr.get("SQL", 0),
        "Excel": curr.get("Excel", 0),
        "HTML": curr.get("HTML_CSS", 0),
        "CSS": curr.get("HTML_CSS", 0),
        "JavaScript": curr.get("JavaScript", 0),
        "Machine Learning": curr.get("ML_Knowledge", 0),
        "Data Analysis": curr.get("Data_Analysis", 0),
        "Pandas": curr.get("Data_Analysis", 0),
        "NumPy": curr.get("Python", 0) and curr.get("Data_Analysis", 0),
        "Statistics": curr.get("Data_Analysis", 0),
        "OOP": curr.get("Java", 0) or curr.get("Cpp", 0) or curr.get("Python", 0),
        "Git": curr.get("Projects", 0) >= 2,
        "Data Structures": curr.get("Cpp", 0) or curr.get("Java", 0)
    }
    student_skills = [k for k, v in skill_mapping.items() if bool(v)]

    all_roles = job_roles_df["Role_Name"].tolist()
    default_role = curr.get("Target_Role", all_roles[0])
    selected_role_name = st.selectbox(
        "Select Target Job Role to Analyze Skill Gap:",
        options=all_roles,
        index=all_roles.index(default_role) if default_role in all_roles else 0
    )

    role_row = job_roles_df[job_roles_df["Role_Name"] == selected_role_name].iloc[0]
    match_detail = calculate_role_match(student_skills, float(curr.get("CGPA", 7.5)), role_row)

    # Top summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Profile Match", f"{match_detail['match_percentage']}%")
    with m2:
        st.metric("Skills You Have", len(match_detail["matching_skills"]))
    with m3:
        st.metric("Missing Skills (Skill Gap)", len(match_detail["missing_skills"]))
    with m4:
        status_text = "Eligible" if match_detail["is_cgpa_eligible"] else "Below Cutoff"
        st.metric("Academic Eligibility", status_text, delta=f"Min CGPA {match_detail['min_cgpa']}")

    st.markdown("---")

    # Visual side-by-side comparison
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("### 🟢 Skills You Have (Matching)")
        if match_detail["matching_skills"]:
            for s in match_detail["matching_skills"]:
                st.success(f"✔ **{s}**")
        else:
            st.info("No matching skills currently detected for this role.")

    with c_right:
        st.markdown("### 🔴 Skills You Need (Skill Gap)")
        if match_detail["missing_skills"]:
            for s in match_detail["missing_skills"]:
                st.error(f"✖ **{s}** (Action Needed)")
        else:
            st.balloons()
            st.success("🎉 Outstanding! You already possess 100% of the core skills required for this role.")

    st.markdown("### 📐 Mathematical Formulation of Matching")
    st.latex(r"\text{Matching Skills} = \text{Student Skills} \cap \text{Required Skills}")
    st.latex(r"\text{Skill Gap} = \text{Required Skills} \setminus \text{Student Skills}")
    st.latex(r"\text{Profile Match (\%)} = \left( \frac{|\text{Matching Skills}|}{|\text{Required Skills}|} \right) \times 100")

# ==============================================================================
# 5. LEARNING RECOMMENDATIONS
# ==============================================================================
elif nav_choice == "📚 Learning Recommendations":
    st.markdown('<div class="main-title">Tailored Learning Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Actionable learning paths, official documentation, and practical milestones for your skill gap.</div>', unsafe_allow_html=True)

    curr = st.session_state.student_profile
    job_roles_df = load_job_roles(paths["job_roles_csv"])

    skill_mapping = {
        "Python": curr.get("Python", 0),
        "C++": curr.get("Cpp", 0),
        "Java": curr.get("Java", 0),
        "SQL": curr.get("SQL", 0),
        "Excel": curr.get("Excel", 0),
        "HTML": curr.get("HTML_CSS", 0),
        "CSS": curr.get("HTML_CSS", 0),
        "JavaScript": curr.get("JavaScript", 0),
        "Machine Learning": curr.get("ML_Knowledge", 0),
        "Data Analysis": curr.get("Data_Analysis", 0),
        "Pandas": curr.get("Data_Analysis", 0),
        "NumPy": curr.get("Python", 0) and curr.get("Data_Analysis", 0),
        "Statistics": curr.get("Data_Analysis", 0),
        "OOP": curr.get("Java", 0) or curr.get("Cpp", 0) or curr.get("Python", 0),
        "Git": curr.get("Projects", 0) >= 2,
        "Data Structures": curr.get("Cpp", 0) or curr.get("Java", 0)
    }
    student_skills = [k for k, v in skill_mapping.items() if bool(v)]

    all_roles = job_roles_df["Role_Name"].tolist()
    target_role_name = curr.get("Target_Role", all_roles[0])
    role_to_study = st.selectbox("Current Selected Role for Roadmap:", options=all_roles, index=all_roles.index(target_role_name) if target_role_name in all_roles else 0)

    role_row = job_roles_df[job_roles_df["Role_Name"] == role_to_study].iloc[0]
    match_detail = calculate_role_match(student_skills, float(curr.get("CGPA", 7.5)), role_row)
    missing_skills = match_detail["missing_skills"]

    if missing_skills:
        st.markdown(f"#### 🎯 Recommended Learning Paths for Missing Skills: {', '.join(f'`{s}`' for s in missing_skills)}")
        recommendations = get_recommendations_for_skills(missing_skills)

        for idx, rec in enumerate(recommendations):
            with st.expander(f"📖 Skill: {rec['skill']} — {rec['title']} ({rec['difficulty']})", expanded=True):
                r_col1, r_col2 = st.columns([2, 1])
                with r_col1:
                    st.write(f"**Platform / Source:** {rec['platform']} | **Type:** {rec['type']}")
                    st.write(f"**Key Topics to Cover:** {rec['key_topics']}")
                    st.write(f"**🎯 Suggested Practice Milestone:** {rec['suggested_milestone']}")
                with r_col2:
                    st.markdown(f"[🔗 Open Official Guide / Resource]({rec['link']})")
                    st.caption("Works offline; link opens standard reference.")
    else:
        st.success("🎉 You meet all technical requirements for this role! Showing Advanced Enrichment Paths:")
        enrichments = get_enrichment_recommendations()
        for rec in enrichments:
            with st.expander(f"🌟 {rec['skill']} — {rec['title']}", expanded=True):
                st.write(f"**Platform:** {rec['platform']} | **Difficulty:** {rec['difficulty']}")
                st.write(f"**Topics:** {rec['key_topics']}")
                st.write(f"**Milestone:** {rec['suggested_milestone']}")
                st.markdown(f"[🔗 Access Resource]({rec['link']})")

# ==============================================================================
# 6. MODEL PERFORMANCE & EVALUATION
# ==============================================================================
elif nav_choice == "📊 Model Performance & Evaluation":
    st.markdown('<div class="main-title">Machine Learning Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Actual empirical evaluation results on the test partition. No hardcoded or faked metrics.</div>', unsafe_allow_html=True)

    if model_bundle is not None:
        eval_results = model_bundle["evaluation_results"]
        best_model_name = model_bundle["best_model_name"]
        
        st.markdown(f"**Selected Best Classifier:** `{best_model_name}` (Selected based on test set F1-Score and Accuracy)")

        # Comparison Table
        table_rows = []
        for model_name, metrics in eval_results.items():
            table_rows.append({
                "Algorithm": model_name,
                "Accuracy": f"{metrics['Accuracy']:.2%}",
                "Precision": f"{metrics['Precision']:.2%}",
                "Recall": f"{metrics['Recall']:.2%}",
                "F1-Score": f"{metrics['F1_Score']:.2%}",
                "Selected Status": "🏆 Best Model" if model_name == best_model_name else "Candidate"
            })
        st.table(pd.DataFrame(table_rows))

        # Model Comparison Chart
        st.markdown("### 📈 Metric Comparison Across Models")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        names = list(eval_results.keys())
        accs = [eval_results[n]["Accuracy"] * 100 for n in names]
        f1s = [eval_results[n]["F1_Score"] * 100 for n in names]

        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, accs, width, label='Accuracy (%)', color='#3B82F6')
        ax.bar(x + width/2, f1s, width, label='F1-Score (%)', color='#10B981')

        ax.set_ylabel('Score (%)')
        ax.set_title('Evaluation on Test Split (20% Holdout)')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 100)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # Confusion Matrix
        st.markdown(f"### 🔲 Confusion Matrix: {best_model_name}")
        cm = np.array(eval_results[best_model_name]["Confusion_Matrix"])
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Not Placed (0)', 'Placed (1)'],
                    yticklabels=['Not Placed (0)', 'Placed (1)'], ax=ax_cm)
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        plt.tight_layout()
        st.pyplot(fig_cm)

        # Feature Importances / Factors Considered
        st.markdown("### 🔍 Factors Considered by the Model (Feature Importance)")
        importances = model_bundle.get("feature_importances", {})
        if importances:
            top_feats = list(importances.keys())[:8]
            top_vals = [importances[k] for k in top_feats]
            
            fig_feat, ax_feat = plt.subplots(figsize=(8, 3.5))
            ax_feat.barh(top_feats[::-1], top_vals[::-1], color='#6366F1')
            ax_feat.set_xlabel("Relative Importance Weight")
            ax_feat.set_title("Top 8 Profile Factors Influencing Placement Prediction")
            plt.tight_layout()
            st.pyplot(fig_feat)
    else:
        st.error("Model artifacts not loaded. Please train the model.")

# ==============================================================================
# 7. ABOUT PROJECT & VIVA GUIDE
# ==============================================================================
elif nav_choice == "ℹ️ About Project & Viva Guide":
    st.markdown('<div class="main-title">Project Documentation & Viva Examination Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quick reference designed for viva questions and project presentation.</div>', unsafe_allow_html=True)

    with st.expander("❓ 1. Why Python and these specific libraries?", expanded=True):
        st.write(
            "- **Python:** Standard industry language for data science and ML with rich ecosystem.\n"
            "- **Pandas & NumPy:** Fast, vectorized tabular data operations and numerical processing.\n"
            "- **Scikit-learn:** Efficient, transparent, and explainable Machine Learning algorithms with standardized evaluation APIs.\n"
            "- **Streamlit:** Python-native framework allowing rapid development of interactive web dashboards without frontend bloat."
        )

    with st.expander("❓ 2. What is the target variable and features used?", expanded=True):
        st.write(
            "- **Target Variable (`Placement_Status`):** Binary classification (1 = Placed / High Likelihood, 0 = Not Placed / Needs Improvement).\n"
            "- **Features:** 18 total profile features comprising Academic standing (CGPA, 10th %, 12th %, Backlogs), Technical skills (Python, SQL, Java, C++, Excel, HTML/CSS, JavaScript, ML, Data Analysis), and Practical/Evaluation metrics (Projects, Internships, Certifications, Aptitude score, Communication score)."
        )

    with st.expander("❓ 3. How does Job Role Matching and Skill Gap work?", expanded=True):
        st.write(
            "It uses transparent Set Theory operations:\n"
            "- Matching Skills = Student Skills ∩ Required Skills\n"
            "- Skill Gap = Required Skills − Student Skills\n"
            "- Profile Match % = (|Matching Skills| / |Required Skills|) * 100\n"
            "This gives zero ambiguity and makes every recommendation explainable to both student and viva examiner."
        )

    with st.expander("❓ 4. What are the limitations and future scope?"):
        st.write(
            "- **Limitations:** Relies on predefined synthetic benchmark data and static skill requirement lists. Does not crawl live hiring portals or assess behavioral video interviews.\n"
            "- **Future Scope:** Resume NLP parsing (extracting skills directly from PDF resumes), dynamic live job board API integration, and adaptive personalized study timetables."
        )

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Academic Project Evaluation Summary:</strong><br>
        • Project: B.Tech CSE Final Year Project<br>
        • Title: Student Placement Analysis and Skill Gap Recommendation System using Machine Learning<br>
        • Execution Mode: Fully local runnable Python Streamlit application.
    </div>
    """, unsafe_allow_html=True)
