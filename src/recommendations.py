"""
recommendations.py
------------------
Generates tailored learning recommendations and structured study pathways
based on the identified skill gaps for targeted job roles.
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd

def get_default_resources_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, "data", "learning_resources.csv")

def load_learning_resources(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads learning resources dataset from CSV.
    """
    path = csv_path or get_default_resources_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Learning resources file not found at: {path}")
    df = pd.read_csv(path)
    return df

PRACTICE_MILESTONES = {
    "Python": "Implement 5 practical mini-projects: a CLI calculator, a CSV data parser, a file organizer script, and basic OOP models.",
    "SQL": "Solve 20 basic and intermediate query challenges on LeetCode / SQLZoo covering JOINs, GROUP BY, HAVING, and subqueries.",
    "Excel": "Create a financial summary dashboard with dynamic Pivot Tables, VLOOKUP/XLOOKUP functions, and KPI conditional formatting.",
    "Statistics": "Work through descriptive statistics, normal distributions, and hypothesis testing (t-test, p-values) with Python SciPy.",
    "Data Analysis": "Complete an exploratory data analysis (EDA) case study on a public Kaggle dataset using Pandas and Matplotlib.",
    "Machine Learning": "Build and evaluate an end-to-end classification pipeline comparing Logistic Regression and Random Forest.",
    "HTML": "Construct 3 semantic responsive landing pages with accessible forms and clean layout hierarchy.",
    "CSS": "Design responsive multi-device layouts utilizing modern Flexbox, CSS Grid, and media query breakpoints.",
    "JavaScript": "Create 3 interactive web apps: a dynamic To-Do list, a real-time search filter, and an API weather dashboard.",
    "C++": "Implement fundamental data structures (Linked Lists, Binary Search Trees, Stacks) in standard modern C++.",
    "Java": "Develop a core OOP console application (Banking / Library Management) demonstrating encapsulation, inheritance, and interfaces.",
    "Data Structures": "Practice algorithmic problem solving: Arrays, Hash Maps, Stacks, Queues, Binary Trees, and Sorting algorithms.",
    "Git": "Initialize a GitHub portfolio repository, practice branching, staging, committing, merging, and resolving conflicts.",
    "OOP": "Refactor a monolithic script into modular classes following Single Responsibility and Open-Closed principles.",
    "Pandas": "Clean a messy real-world CSV dataset: handle missing values, reshape columns, and generate summary statistics.",
    "NumPy": "Master multi-dimensional array operations, matrix multiplication, broadcasting, and vectorization.",
    "Linux": "Familiarize yourself with Linux CLI: navigate directories, manage file permissions (chmod), and write simple Bash scripts.",
    "React": "Build a single-page task manager with component state management (useState, useEffect) and reusable component cards.",
    "Database Management": "Design an Entity-Relationship (ER) diagram for an e-commerce platform and normalize it to 3NF.",
    "Cloud Basics": "Deploy a simple static website or containerized Python application on a free cloud tier (AWS / Render / Streamlit Cloud).",
    "Networking": "Understand the TCP/IP stack, DNS lookups, HTTP methods (GET, POST, PUT, DELETE), and status codes.",
    "Backup Recovery": "Practice exporting and restoring a MySQL database dump using mysqldump and source commands."
}

def get_recommendations_for_skills(
    missing_skills: List[str],
    resources_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Finds learning resources and practical milestones for each missing skill.
    """
    df = resources_df if resources_df is not None else load_learning_resources()
    recommendations = []
    
    # Normalize skill names
    normalized_missing = [s.strip() for s in missing_skills if s.strip()]
    
    for skill in normalized_missing:
        # Match case-insensitively
        matches = df[df["Skill"].str.lower() == skill.lower()]
        
        milestone = PRACTICE_MILESTONES.get(
            skill,
            f"Review official documentation and complete 3 hands-on practical exercises focused on {skill}."
        )
        
        if not matches.empty:
            for _, row in matches.iterrows():
                recommendations.append({
                    "skill": row["Skill"],
                    "category": row.get("Category", "General"),
                    "title": row.get("Resource_Title", f"{skill} Learning Guide"),
                    "type": row.get("Resource_Type", "Online Course / Guide"),
                    "platform": row.get("Platform_Or_Source", "Official Docs"),
                    "key_topics": row.get("Key_Topics_To_Cover", "Core concepts and hands-on practice"),
                    "link": row.get("Link", "https://www.google.com"),
                    "difficulty": row.get("Difficulty_Level", "Beginner"),
                    "suggested_milestone": milestone
                })
        else:
            # Fallback recommendation if skill is not explicitly in CSV
            recommendations.append({
                "skill": skill,
                "category": "Technical Skill",
                "title": f"Fundamentals of {skill}",
                "type": "Documentation & Exercises",
                "platform": "Official Documentation / GeeksforGeeks",
                "key_topics": f"Foundational concepts, syntax, best practices, and exercises for {skill}",
                "link": f"https://www.google.com/search?q={skill}+tutorial+documentation",
                "difficulty": "Beginner to Intermediate",
                "suggested_milestone": milestone
            })
            
    return recommendations

def get_enrichment_recommendations() -> List[Dict[str, Any]]:
    """
    Provides advanced enrichment recommendations when a student already has
    all required skills (0 skill gap).
    """
    return [
        {
            "skill": "System Design & Architecture",
            "category": "Advanced Engineering",
            "title": "System Design Primer & Scalable Architectures",
            "type": "Comprehensive Guide / Case Studies",
            "platform": "GitHub (donnemartin/system-design-primer)",
            "key_topics": "Microservices, Load Balancing, Caching, Database Sharding, Asynchronous Processing",
            "link": "https://github.com/donnemartin/system-design-primer",
            "difficulty": "Advanced",
            "suggested_milestone": "Design the high-level architecture for a URL shortener or rate limiter."
        },
        {
            "skill": "Technical Interview Preparation",
            "category": "Career Readiness",
            "title": "Data Structures & Algorithms Problem Solving",
            "type": "Interactive Coding Platform",
            "platform": "LeetCode / HackerRank",
            "key_topics": "Two Pointers, Sliding Window, Dynamic Programming, Graph Traversals (BFS/DFS)",
            "link": "https://leetcode.com/explore/",
            "difficulty": "Intermediate to Advanced",
            "suggested_milestone": "Solve the Top 50 Interview Questions on LeetCode with clean time/space complexity."
        },
        {
            "skill": "Open Source Contributions",
            "category": "Portfolio Enhancement",
            "title": "Contributing to Open Source Projects",
            "type": "Practical Guide",
            "platform": "Open Source Guides / GitHub",
            "key_topics": "Issue tracking, Forking, Pull Requests, Code Reviews, Writing Documentation",
            "link": "https://opensource.guide/how-to-contribute/",
            "difficulty": "Intermediate",
            "suggested_milestone": "Submit a documentation or bug-fix pull request to a popular open-source repository."
        }
    ]
