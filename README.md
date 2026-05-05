# 🎓 Comparative Data Mining Analysis for Student Dropout Prediction and Risk Profiling

## 📌 Introduction

Student dropout is a major challenge in higher education institutions worldwide. Universities invest significant resources in student admissions and academic support, yet many students fail to complete their degrees. This leads to financial loss for institutions and negative academic and career outcomes for students.

Traditional approaches to identifying at-risk students are often reactive, meaning interventions occur too late. With the advancement of data mining and machine learning techniques, it is now possible to proactively identify students at risk of dropping out.

This project applies a comprehensive data mining framework to analyze student data and address dropout prediction. The goal is not only to build predictive models but also to uncover meaningful patterns and student risk profiles that can help institutions take early and informed actions.

---

## 📊 Dataset Description

The dataset used in this project is:

**University Student Dropout: A Longitudinal Dataset of Demographic, Socioeconomic, and Academic Indicators**

- 📍 Source: Universitat Politècnica de València (UPV), Spain
- 📰 Publication: MDPI *Data* (October 2025, CC BY 4.0)
- 🔗 DOI: https://doi.org/10.3390/data10100162
- 📦 Total Records: 464,739
- 👨‍🎓 Unique Students: 39,364
- 📚 Courses: 4,989 across 163 degree programs
- 🎯 Features: 77 variables
- ⏳ Time Coverage: 3 academic years (COVID-affected periods excluded)
- 🗂️ Structure: One row represents a **student × course × academic year**

### 🎯 Target Variable
- **Dropout Indicator**
  - `1` → Dropped out  
  - `0` → Continued / Completed  

### ⭐ Key Characteristics

- Large-scale dataset suitable for robust analysis  
- Longitudinal data capturing student progression over time  
- Includes academic, behavioral, and socioeconomic features  
- Designed specifically for dropout prediction research  

---

## 🧩 Feature Categories

The dataset consists of **77 features grouped into five major categories**:

### 1. Admission & Enrollment Features
- Admission score  
- Application type  
- Degree program  
- Enrollment status (full-time / part-time)  
- Academic year  

👉 Helps understand how students enter the institution.

---

### 2. Demographic & Socioeconomic Features
- Age  
- Gender  
- Nationality  
- Family income  
- Scholarship status  
- Geographic origin  

👉 Captures external factors influencing student success.

---

### 3. Academic Performance Features
- GPA  
- Course grades  
- Pass/fail status  
- Credits attempted vs passed  
- Academic progression rate  

👉 Strong indicators of academic success and dropout risk.

---

### 4. Digital Engagement Features
- LMS logins  
- Assignment submissions  
- Resource access  
- Forum participation  

👉 Reflects student engagement with learning platforms.

---

### 5. Physical Campus Presence
- Wi-Fi access logs  
- Campus activity frequency  

👉 Serves as a proxy for attendance and physical engagement.

---

## 🔍 Research Problems

This project addresses three key data mining problems:

---

### 🧠 Problem 1 — Dropout Prediction (Classification)

**Goal:** Predict whether a student will drop out or continue.

- Type: Supervised Learning (Binary Classification)  
- Input: Academic, demographic, and engagement features  

**Challenges:**
- Class imbalance  
- Complex feature relationships  
- Behavioral variability across students  

**Objective:**
Develop an early warning system to identify at-risk students before dropout occurs.

---

### 📊 Problem 2 — Student Risk Profiling (Clustering)

**Goal:** Identify groups of students with similar characteristics.

- Type: Unsupervised Learning  

**Purpose:**
Discover hidden patterns in student behavior without using labels.

**Expected Outcomes:**
- High-performing students  
- At-risk disengaged students  
- Moderate-risk groups  

**Objective:**
Provide interpretable student segments for better decision-making.

---

### 🔗 Problem 3 — Pattern Discovery (Association Rule Mining)

**Goal:** Identify combinations of features associated with dropout.

- Type: Association Rule Mining  

**Example Insight:**
Low LMS activity + Low GPA + Part-time enrollment → High dropout risk  

**Metrics Used:**
- Support  
- Confidence  
- Lift  

**Objective:**
Generate human-readable rules that can support institutional decision-making.

---

## 🛠️ Project Structure

```
.
├── README.md
├── classification.py          # Person 2 — Supervised learning (LR / DT / RF / KNN / SVM)
├── clustering.py              # Person 3 — K-Means + Agglomerative (Ward)
├── association_rules.py       # Person 4 — Apriori + FP-Growth
├── data/
│   ├── dataset_2018_hash.csv
│   ├── dataset_2021_hash.csv
│   ├── dataset_2022_hash.csv
│   └── dataset_info.txt
├── docs/
│   ├── introduction.md
│   ├── dataset_description.md
│   ├── research_problems.md
│   └── final_insights.md      # Person 4 — recommendations + advisor playbook
└── results/
    ├── classification/        # confusion matrices + summary CSV + RF importances
    ├── clustering/            # elbow, dendrogram, PCA scatter, profiles, summary
    └── association_rules/     # frequent itemsets, top-20 by lift, dropout rules
```

## ⚙️ How to run

```bash
pip install -r requirements.txt   # or: pip install scikit-learn imbalanced-learn mlxtend matplotlib scipy pandas
python classification.py
python clustering.py
python association_rules.py
```

---

## 🚀 Expected Outcomes

- Accurate prediction of student dropout using classification models  
- Identification of key factors influencing dropout  
- Discovery of meaningful student groups through clustering  
- Extraction of actionable patterns using association rules  
- Data-driven insights for improving student retention strategies  

---

## 📌 Conclusion

This project demonstrates how data mining techniques can be used not only to predict student dropout but also to understand the underlying causes. By combining classification, clustering, and association rule mining, the project provides both predictive power and interpretability, enabling institutions to take proactive and informed actions.

---

## 👥 Team Members

- Chaya Sri K  
- Joshua Godwin  
- Bharath Raahul Murugesan  
- Bansari Yadav  

---

