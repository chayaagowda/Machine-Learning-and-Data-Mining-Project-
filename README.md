# 🎓 Comparative Data Mining Analysis for Student Dropout Prediction and Risk Profiling

## 📌 Introduction

Student dropout is a major challenge in higher education institutions worldwide. Universities invest significant resources in student admissions and academic support, yet many students fail to complete their degrees. This leads to financial loss for institutions and negative academic and career outcomes for students.

Traditional approaches to identifying at-risk students are often reactive, meaning interventions occur too late. With the advancement of data mining and machine learning techniques, it is now possible to proactively identify students at risk of dropping out.

This project applies a comprehensive data mining framework to analyze student data and address dropout prediction. The goal is not only to build predictive models but also to uncover meaningful patterns and student risk profiles that can help institutions take early and informed actions.

---

## 📊 Dataset Description

The dataset used in this project is:

**University Student Dropout: A Longitudinal Dataset of Demographic, Socioeconomic, and Academic Indicators**

- 📍 Source: UCI / UPV (Universitat Politècnica de València, Spain)  
- 📅 Published: October 2025  
- 📦 Total Records: 464,739  
- 👨‍🎓 Unique Students: 39,364  
- 📚 Courses: 4,989  
- 🎯 Features: 77 variables  
- ⏳ Time Coverage: 3 academic years  
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
