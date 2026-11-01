# Student Placement Analysis and Skill Gap Recommendation System using Machine Learning

A transparent, explainable, and production-ready academic decision-support web application for B.Tech Computer Science and Engineering students.

---

## 1. Project Overview

During campus placements, students frequently face uncertainty regarding their placement preparedness, suitable industry roles, and specific skill deficiencies. Existing systems usually provide only a black-box binary classification (*Placed* vs *Not Placed*) without explaining **why** that outcome was predicted or **how** the student can bridge the gap.

This project addresses that research gap by bridging:
1. **Machine Learning Placement Possibility Prediction** (trained on benchmark student profiles).
2. **Transparent Profile Factor Analysis** (strengths and improvement areas).
3. **Predefined Job Role & Company Matching** (set-theoretic profile compatibility).
4. **Skill Gap Identification** ($\text{Required Skills} \setminus \text{Student Skills}$).
5. **Targeted Learning Recommendations** (curated documentation, interactive tutorials, and coding milestones).

> **Important Academic Disclaimer:** This system is an advisory decision-support tool. It does **not** promise or guarantee employment. All predictions are statistical estimates derived from historical patterns.

---

## 2. Key Features

- **Interactive Student Profile Evaluation:** Input academic percentages (10th, 12th, B.Tech CGPA), active backlogs, technical skill proficiencies, practical projects, internships, certifications, and aptitude/communication scores.
- **Explainable Machine Learning:** Compares **Logistic Regression**, **Decision Tree**, and **Random Forest Classifiers** on test partitions. Evaluates real metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- **Set-Theoretic Skill Gap Analysis:** Mathematically models matching skills via intersection ($\cap$) and missing skills via difference ($\setminus$).
- **Multi-Role Industry Compatibility:** Evaluates compatibility across 8 industry roles (Data Analyst, Python Developer, Web Developer, ML Intern, Software Developer, Full Stack Developer, DBA Trainee, Cloud Trainee).
- **Curated Learning Roadmaps:** Directs students to high-quality official guides (Python.org, MDN Web Docs, W3Schools, Kaggle, Khan Academy) with practical coding milestones.
- **Embedded Viva Examination Guide:** Comprehensive explanations for college examiners and viva panels.

---

## 3. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Programming Language** | Python 3.13 | Core backend and pipeline logic |
| **Data Processing** | Pandas, NumPy | Data cleaning, validation, and vector math |
| **Machine Learning** | Scikit-learn, Joblib | Model training, scaling, evaluation, and persistence |
| **Visualization** | Matplotlib, Seaborn | Confusion matrices, performance comparison charts |
| **User Interface / Dashboard** | Streamlit | Responsive multi-page web application |
| **Storage / Data Layer** | CSV & SQLite | Clean flat files with optional embedded relational SQL |

---

## 4. Project Structure

```text
placement_analysis/
│
├── app.py                      # Main Streamlit web dashboard application
├── requirements.txt            # Project Python dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules
│
├── data/                       # Datasets
│   ├── students.csv            # Synthetic benchmark student cohort (700 records)
│   ├── job_roles.csv           # Predefined industry job roles & requirements
│   ├── learning_resources.csv  # Curated skill learning resources & tutorials
│   └── placement_records.db    # Embedded SQLite database (relational sync)
│
├── models/                     # Trained Machine Learning artifacts
│   ├── placement_model.pkl     # Best trained classifier, scaler, and metadata
│   └── model_metrics.json      # Human-readable evaluation metrics & feature importances
│
├── src/                        # Modular Python source code
│   ├── __init__.py             # Package marker
│   ├── generate_dataset.py     # Reproducible synthetic dataset generator
│   ├── data_preprocessing.py   # Data loader, schema validation, and scaling
│   ├── model_training.py       # ML training, cross-model comparison, and export
│   ├── prediction.py           # Inference engine and explainable factor analysis
│   ├── skill_matching.py       # Set-theoretic job matching and eligibility logic
│   ├── recommendations.py      # Targeted learning paths and practical milestones
│   └── db_handler.py           # Data access layer (CSV & SQLite abstraction)
│
├── notebooks/                  # Academic walkthrough notebooks
│   ├── placement_analysis.ipynb# Full EDA and ML workflow walkthrough
│   └── create_nb.py            # Automated notebook builder
│
└── tests/                      # Automated unit and integration test suite
    └── test_pipeline.py        # Test cases for strong/weak profiles, gaps, and edge cases
```

---

## 5. Machine Learning Methodology

### 5.1 Dataset Features
- **Academic Factors:** `CGPA`, `Tenth_Percentage`, `Twelfth_Percentage`, `Backlogs`
- **Technical Skills:** `Python`, `Cpp`, `Java`, `SQL`, `Excel`, `HTML_CSS`, `JavaScript`, `ML_Knowledge`, `Data_Analysis`
- **Experience & Assessment:** `Projects`, `Internships`, `Certifications`, `Aptitude_Score`, `Communication_Score`
- **Target Variable:** `Placement_Status` (1 = Placed / High Likelihood, 0 = Needs Improvement)

### 5.2 Preprocessing & Scaling
Continuous numerical features (`CGPA`, `Tenth_Percentage`, `Twelfth_Percentage`, `Aptitude_Score`, `Communication_Score`) are scaled using `StandardScaler` fitted strictly on training data ($z = \frac{x - \mu}{\sigma}$).

### 5.3 Model Evaluation
Three explainable algorithms are trained on an 80-20 stratified train-test split:
1. **Logistic Regression** (Linear decision boundary, explainable feature odds)
2. **Decision Tree Classifier** (Hierarchical rule splits, non-linear boundaries)
3. **Random Forest Classifier** (Ensemble of bagged trees, high generalization)

The highest F1-score model is automatically selected and serialized.

---

## 6. Mathematical Formulation

### 6.1 Set-Theoretic Skill Matching
$$\text{Matching Skills} = \text{Student Skills} \cap \text{Required Skills}$$
$$\text{Skill Gap (Missing)} = \text{Required Skills} \setminus \text{Student Skills}$$
$$\text{Profile Match (\%)} = \left( \frac{|\text{Matching Skills}|}{|\text{Required Skills}|} \right) \times 100$$

### 6.2 Academic Cutoff Verification
$$\text{Eligibility} = \begin{cases} 
\text{Eligible}, & \text{if } \text{CGPA}_{\text{student}} \ge \text{Min CGPA}_{\text{role}} \\
\text{Below Cutoff}, & \text{otherwise}
\end{cases}$$

---

## 7. Setup & Execution Instructions

### Prerequisites
- Windows 10/11
- Python 3.13 installed
- VS Code (recommended)

### Step 1: Open Terminal in VS Code
Open VS Code, press `Ctrl + ~` to open the integrated terminal, and navigate to the project directory:
```powershell
cd "c:\Users\vakan\OneDrive\Documents\placement_analysis"
```

### Step 2: Create and Activate Virtual Environment
```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Train Machine Learning Models
```powershell
python src/model_training.py
```
This trains the models on `data/students.csv`, evaluates them on holdout test samples, and saves the best model bundle to `models/placement_model.pkl`.

### Step 5: Run Automated Test Suite
```powershell
python -m unittest tests/test_pipeline.py
```

### Step 6: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
Streamlit will launch locally in your web browser at `http://localhost:8501`.

---

## 8. Viva Examination Guide & Answers

- **Q1: Why use Machine Learning instead of hardcoded if-else rules?**
  *Answer:* Hardcoded cutoffs fail to capture nuanced trade-offs (e.g. a student with a slightly lower CGPA but exceptional project portfolio and high aptitude). Machine learning algorithms learn multivariate non-linear decision boundaries from historical data.

- **Q2: Why is Random Forest or Logistic Regression preferred over Deep Learning here?**
  *Answer:* Tabular data with continuous scores and binary indicators performs reliably on tree-based ensembles and generalized linear models. Deep learning requires tens of thousands of samples, lacks native explainability, and is prone to overfitting small tabular datasets.

- **Q3: How are missing skills identified?**
  *Answer:* We represent student proficiencies and role requirements as discrete mathematical sets. The skill gap is the set difference between required skills and student skills: $\text{Required} \setminus \text{Student}$.

- **Q4: What happens if a student has active backlogs?**
  *Answer:* The model penalizes active backlogs based on historical recruitment criteria where companies impose a zero-backlog eligibility filter. The dashboard flags backlogs under "Improvement Areas".

---

## 9. Limitations & Future Scope

### Limitations
1. Benchmark dataset contains synthetic records calibrated to realistic distributions rather than live proprietary university archives.
2. Job roles and sample hiring companies are predefined templates, not live scraped listings.
3. Does not evaluate soft skills via audio/video analysis.

### Future Scope
1. **Resume Parser (NLP):** Automatically extracting skills directly from PDF/DOCX resumes using spaCy or regular expressions.
2. **Live Job Board APIs:** Integrating real-time vacancy listings from corporate APIs.
3. **Personalized Timetables:** Automatically generating 4-week weekly study schedules based on student graduation date.

---

## 10. License & Academic Attribution
Developed as an academic capstone project for B.Tech Computer Science and Engineering.
