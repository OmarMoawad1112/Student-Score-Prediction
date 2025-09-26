# 🎓 Student Score Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green.svg)

## 📌 Project Overview
This project analyzes the **Student Performance Factors** dataset to explore how different factors (study hours, attendance, parental involvement, etc.) impact students' academic performance.  
The main objective is to **predict final exam scores** using machine learning regression models.

---

## 📂 Dataset
The dataset comes from [Kaggle: Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors).  

- **Rows:** ~6,600 students  
- **Columns:** ~20 features + target (`Exam_Score`)  
- Features include academic, social, and demographic factors such as:
  - Hours Studied  
  - Attendance  
  - Previous Scores  
  - Parental Involvement  
  - Sleep Hours  
  - Extracurricular Activities  
  - Family Income, School Type, etc.  

Target variable:  
- `Exam_Score` → numerical final exam score of the student  

---

## 🔑 Steps in the Notebook
1. **Data Loading & Cleaning**
   - Loaded dataset (`StudentPerformanceFactors.csv`)
   - Removed invalid scores (e.g., >100)
   - Handled missing values and checked distributions

2. **Exploratory Data Analysis (EDA)**
   - Histograms, boxplots, scatter plots
   - Correlation heatmap
   - Outlier detection and handling

3. **Feature Engineering**
   - Encoding categorical variables with `pd.get_dummies`
   - Normalization/Scaling
   - Log transformation for skewed data

4. **Model Training**
   - Linear Regression
   - Polynomial Regression
   - Regularization techniques (Ridge/Lasso, if tested)
   - Cross-validation for robust performance estimates

5. **Evaluation**
   - Metrics: R² Score, RMSE
   - Comparison of train/test performance
   - Visualization of predicted vs. actual scores

---

## 📊 Results & Insights
- **Strong predictors:** study hours, attendance, and previous scores  
- **Preprocessing helped:** log transformations and clipping outliers improved model generalization  
- **Performance:** final models achieved **R² between 0.75 – 0.90** depending on preprocessing and model type  
- Scatterplots showed a strong alignment between predicted and actual scores  

---

## 🚀 Tech Stack
- **Python** (3.9+)  
- **Libraries:**  
  - pandas, numpy  
  - matplotlib, seaborn  
  - scikit-learn  

---

## ▶️ How to Run
1. Clone this repo:  
   ```bash
   git clone https://github.com/your-username/student-score-prediction.git
   cd student-score-prediction
