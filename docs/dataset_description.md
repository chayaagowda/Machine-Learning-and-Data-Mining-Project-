## Dataset Description

The dataset used in this project is the **University Student Dropout: A Longitudinal Dataset of Demographic, Socioeconomic, and Academic Indicators**. It is designed to support research in student retention, dropout prediction, and educational data mining.

### Source

- Provider: Universitat Politècnica de València (UPV), Spain
- Published October 2025 in MDPI *Data* (open access, CC BY 4.0)
- Paper DOI: https://doi.org/10.3390/data10100162
- Title: *University Student Dropout: A Longitudinal Dataset of Demographic, Socioeconomic, and Academic Indicators*

### Dataset Size

- Total Records: **464,739**
- Unique Students: **39,364**
- Courses: **4,989**
- Features: **77 variables**
- Time Span: **3 academic years**

### Dataset Structure

Each row in the dataset represents a combination of:

- One student  
- One enrolled course  
- One academic year  

This longitudinal structure allows tracking of student progress over time and across multiple courses.

### Target Variable

The primary target variable used in this project is:

- **Dropout Status**
  - `1` = Student dropped out
  - `0` = Student continued or completed studies

### Feature Categories

The 77 features are grouped into the following categories:

#### 1. Admission and Enrollment Features
- Admission score  
- Application type  
- Degree program  
- Enrollment mode  
- Academic year  

#### 2. Demographic and Socioeconomic Features
- Age  
- Gender  
- Nationality  
- Scholarship status  
- Family background  

#### 3. Academic Performance Features
- GPA  
- Course grades  
- Credits attempted  
- Credits passed  
- Progression rate  

#### 4. Digital Engagement Features
- LMS logins  
- Assignment submissions  
- Resource access frequency  
- Online participation  

#### 5. Campus Presence Features
- Wi-Fi connection logs  
- Physical campus activity records  

### Importance of the Dataset

This dataset is highly valuable because it:

- Contains real-world university student data  
- Combines academic, behavioral, and socioeconomic indicators  
- Supports predictive modeling and clustering tasks  
- Enables early identification of at-risk students  
- Helps institutions improve retention strategies  

### Challenges in the Dataset

Some common challenges include:

- Large dataset size requiring preprocessing  
- Missing or noisy values  
- Class imbalance in dropout cases  
- High dimensionality with 77 features  

Overall, the dataset provides a strong foundation for applying data mining techniques to understand and predict student dropout behavior.
