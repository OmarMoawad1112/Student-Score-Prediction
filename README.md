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

- **Study hours** and **attendance**  were the strongest predictors of exam performance.

- Outliers were handled with the **IQR method**: numeric features (Hours_Studied, Attendance, Previous_Scores, Tutoring_Sessions) and the target (Exam_Score) were clipped to reduce noise and improve model stability.

- **Hypothesis testing (ANOVA & t-test)** showed that **Gender** and **School_Type** had no statistically significant effect on exam scores (p-value > 0.05).

- **Linear Regression** performed well with an **R² ≈ 0.86** on the test set, showing strong predictive power.

- **Polynomial Regression** (degree 2) achieved an **R² ≈ 0.85** on the test set, which is very close to the linear model’s performance, confirming that the relationship between predictors and target is mostly linear.

Future improvements could include using Ridge/Lasso regularization, adding interaction terms, or testing tree-based models like Random Forest or Gradient Boosting.
  
---

## ▶️ How to Run
1. Clone this repo:  
   ```bash
   git clone https://github.com/your-username/student-score-prediction.git
   cd student-score-prediction
