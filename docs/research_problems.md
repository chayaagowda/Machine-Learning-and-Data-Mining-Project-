## Research Problems

This project focuses on applying data mining techniques to address the issue of student dropout in higher education. Rather than relying on only one method, the project explores three major research problems: **classification, clustering, and association rule mining**. These approaches provide both predictive accuracy and meaningful insights for decision-makers.

---

## Problem 1: Student Dropout Prediction (Classification)

### Objective

The first research problem is to predict whether a student is likely to **drop out** or **continue/completed studies** based on historical data.

### Type of Learning

- Supervised Learning  
- Binary Classification  

### Input Features

The prediction model uses multiple feature categories such as:

- Academic performance  
- Admission information  
- Demographic factors  
- Socioeconomic background  
- LMS engagement data  
- Campus presence records  

### Expected Output

- `1` = Student likely to drop out  
- `0` = Student likely to continue/completed  

### Importance

Early prediction helps institutions identify at-risk students before dropout occurs, allowing timely academic counseling, mentoring, and support programs.

### Possible Algorithms

- Decision Tree  
- Random Forest  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)

### Challenges

- Class imbalance (fewer dropout students)  
- Missing values  
- Complex relationships between variables  
- Changing student behavior over time  

---

## Problem 2: Student Risk Profiling (Clustering)

### Objective

The second research problem is to group students into clusters based on similarities in academic behavior, engagement, and performance.

### Type of Learning

- Unsupervised Learning  
- Clustering Analysis  

### Purpose

Unlike classification, clustering does not use labels. It discovers hidden structures and naturally occurring student groups.

### Expected Clusters

Examples of clusters may include:

- High-performing and highly engaged students  
- Low-performing and disengaged students  
- Average students with irregular attendance  
- Financially challenged but academically strong students  

### Importance

Clustering helps institutions design personalized interventions for different student groups rather than using one solution for all students.

### Possible Algorithms

- K-Means Clustering  
- Hierarchical Clustering  
- DBSCAN  

### Benefits

- Better student segmentation  
- Improved academic advising  
- Customized retention strategies  
- Identification of hidden risk groups  

---

## Problem 3: Dropout Pattern Discovery (Association Rule Mining)

### Objective

The third research problem is to identify combinations of factors that frequently occur together and are associated with dropout.

### Type of Learning

- Unsupervised Learning  
- Pattern Mining / Association Rule Mining  

### Example Rules

- Low GPA + Low LMS Activity → Dropout  
- Part-time Enrollment + Low Attendance → Dropout  
- Failed Core Courses + Low Credits Passed → High Risk  

### Evaluation Metrics

- **Support**: Frequency of rule occurrence  
- **Confidence**: Probability that rule leads to dropout  
- **Lift**: Strength of association compared to random chance  

### Importance

Association rules provide clear and interpretable knowledge that administrators can directly use for decision-making.

### Benefits

- Easy-to-understand insights  
- Explains why dropout happens  
- Supports policy planning  
- Improves transparency of analytics  

---

## Overall Research Contribution

This project combines three complementary approaches:

- **Classification** predicts who may drop out  
- **Clustering** identifies student risk groups  
- **Association Rules** explain why dropout happens  

By integrating these methods, the project provides both predictive power and actionable insights for improving student retention.

---

## Final Goal

The ultimate goal of this research is to help educational institutions move from reactive decisions to proactive student success strategies using data-driven intelligence.
