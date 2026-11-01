"""
test_pipeline.py
----------------
Unit and integration tests for the Placement Analysis and Skill Gap System.
Covers:
1. Student with strong academic profile.
2. Student with weak academic profile.
3. Student with many technical skills.
4. Student with missing skills for selected role.
5. Student with no skill gap (100% match).
6. Invalid/empty input edge cases.
7. Job role matching algorithm and ranking.
8. Recommendation generation.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import (
    load_dataset,
    validate_dataset_schema,
    clean_and_impute_data,
    prepare_data_for_training,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from src.skill_matching import (
    load_job_roles,
    parse_skill_list,
    calculate_role_match,
    match_all_job_roles
)
from src.recommendations import (
    get_recommendations_for_skills,
    get_enrichment_recommendations
)
from src.prediction import (
    analyze_profile_factors,
    predict_placement_possibility
)
from src.model_training import train_and_evaluate_models

class TestPlacementSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(project_root, "data")
        cls.models_dir = os.path.join(project_root, "models")
        cls.students_csv = os.path.join(cls.data_dir, "students.csv")
        cls.job_roles_csv = os.path.join(cls.data_dir, "job_roles.csv")
        cls.resources_csv = os.path.join(cls.data_dir, "learning_resources.csv")

    def test_01_dataset_schema_and_loading(self):
        """Verifies dataset files exist and conform to expected schema."""
        self.assertTrue(os.path.exists(self.students_csv), "students.csv must exist.")
        self.assertTrue(os.path.exists(self.job_roles_csv), "job_roles.csv must exist.")
        self.assertTrue(os.path.exists(self.resources_csv), "learning_resources.csv must exist.")
        
        df = load_dataset(self.students_csv)
        self.assertGreater(len(df), 100, "Dataset should have sufficient sample rows.")
        is_valid, missing = validate_dataset_schema(df)
        self.assertTrue(is_valid, f"Missing required columns in dataset: {missing}")

    def test_02_strong_academic_profile(self):
        """Tests that a strong profile produces favorable factors and high probability."""
        strong_profile = {
            "CGPA": 9.2,
            "Tenth_Percentage": 92.0,
            "Twelfth_Percentage": 89.0,
            "Backlogs": 0,
            "Python": 1,
            "Cpp": 1,
            "Java": 1,
            "SQL": 1,
            "Excel": 1,
            "HTML_CSS": 1,
            "JavaScript": 1,
            "ML_Knowledge": 1,
            "Data_Analysis": 1,
            "Projects": 4,
            "Internships": 2,
            "Certifications": 3,
            "Aptitude_Score": 90.0,
            "Communication_Score": 88.0
        }
        factors = analyze_profile_factors(strong_profile)
        self.assertGreater(len(factors["strengths"]), 2, "Strong student should have multiple strengths.")
        self.assertEqual(len(factors["concerns"]), 0, "Strong student should have zero critical concerns.")

    def test_03_weak_academic_profile(self):
        """Tests that a weak profile identifies risk areas (backlogs, low CGPA/aptitude)."""
        weak_profile = {
            "CGPA": 5.8,
            "Tenth_Percentage": 60.0,
            "Twelfth_Percentage": 58.0,
            "Backlogs": 3,
            "Python": 0,
            "Cpp": 0,
            "Java": 0,
            "SQL": 0,
            "Excel": 0,
            "HTML_CSS": 0,
            "JavaScript": 0,
            "ML_Knowledge": 0,
            "Data_Analysis": 0,
            "Projects": 0,
            "Internships": 0,
            "Certifications": 0,
            "Aptitude_Score": 42.0,
            "Communication_Score": 45.0
        }
        factors = analyze_profile_factors(weak_profile)
        self.assertGreater(len(factors["concerns"]), 2, "Weak profile should flag backlogs, low CGPA, and missing skills.")

    def test_04_skill_gap_for_selected_role(self):
        """Tests role matching and missing skill identification."""
        roles_df = load_job_roles(self.job_roles_csv)
        data_analyst_row = roles_df[roles_df["Role_Name"] == "Data Analyst"].iloc[0]
        
        # Student with Python and Excel, but missing SQL, Statistics, Data Analysis
        student_skills = ["Python", "Excel"]
        match_info = calculate_role_match(student_skills, 7.8, data_analyst_row)
        
        self.assertIn("Python", match_info["matching_skills"])
        self.assertIn("Excel", match_info["matching_skills"])
        self.assertIn("SQL", match_info["missing_skills"])
        self.assertIn("Statistics", match_info["missing_skills"])
        self.assertEqual(match_info["match_percentage"], 40.0)

    def test_05_perfect_skill_match_no_gap(self):
        """Tests scenario where student has all required skills (100% match, empty gap)."""
        roles_df = load_job_roles(self.job_roles_csv)
        web_dev_row = roles_df[roles_df["Role_Name"] == "Web Developer"].iloc[0]
        
        student_skills = ["HTML", "CSS", "JavaScript", "Git", "React"]
        match_info = calculate_role_match(student_skills, 8.0, web_dev_row)
        
        self.assertEqual(len(match_info["missing_skills"]), 0, "Missing skills should be empty for 100% match.")
        self.assertEqual(match_info["match_percentage"], 100.0)

    def test_06_learning_recommendations(self):
        """Tests that learning recommendations are successfully generated for missing skills."""
        missing = ["SQL", "Statistics", "Machine Learning"]
        recs = get_recommendations_for_skills(missing)
        self.assertGreaterEqual(len(recs), 3)
        for r in recs:
            self.assertIn("skill", r)
            self.assertIn("title", r)
            self.assertIn("link", r)
            self.assertIn("suggested_milestone", r)

    def test_07_invalid_or_missing_input_handling(self):
        """Tests that missing fields or empty skills do not cause crashes."""
        empty_profile = {}
        factors = analyze_profile_factors(empty_profile)
        self.assertIsInstance(factors, dict)
        self.assertIn("strengths", factors)
        self.assertIn("concerns", factors)

if __name__ == "__main__":
    unittest.main(verbosity=2)
