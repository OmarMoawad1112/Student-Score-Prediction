# Data handling
import pandas as pd
import numpy as np
import math

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning (Scikit-learn)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Hypothesis Testing
from scipy.stats import f_oneway, ttest_ind





print('1-Load the dataset and quick overview.\n')
df = pd.read_csv("../data/StudentPerformanceFactors.csv")
print(df.head().to_string())
print('==============================================================================================================')

print(df.info())

print('==============================================================================================================')

print(df.describe().to_string())

print('==============================================================================================================')

df = df[df['Exam_Score'] <= 100] # keeping only valid exam scores
print(f"number of duplicated rows = {df.duplicated().sum()}")

print('==============================================================================================================')

print("number of missing values in each column\n")
print(df.isna().sum())

# replacing missing values with the most frequent value (mode) in each column.
df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])

print('==============================================================================================================')

print("checking data types\n")
print(df.dtypes)

# ==============================================================================================================

# Univariate Analysis:** Visualize distributions of numerical features using histograms and box plots.
numerical_cols = df.select_dtypes(include=['int64']).columns.to_list()

# Automatically calculate rows and columns
n_cols = 3  # number of plots per row
n_rows = math.ceil(len(numerical_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()  # flatten in case of multiple rows

for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, bins=20, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Remove any empty subplots (if total plots < n_rows*n_cols)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ==============================================================================================================


fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()  # flatten in case of multiple rows

for i, col in enumerate(numerical_cols):
    sns.boxplot(x=df[col], ax=axes[i])

# Remove any empty subplots (if total plots < n_rows*n_cols)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.show()

# ==============================================================================================================

# Bivariate Analysis: Explore relationships between study hours and exam scores using scatter plots.
scatter_numerical_cols = numerical_cols.copy()
scatter_numerical_cols.remove('Exam_Score')

# Automatically calculate rows and columns
n_cols = 3  # number of plots per row
n_rows = math.ceil(len(scatter_numerical_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()  # flatten in case of multiple rows

for i, col in enumerate(scatter_numerical_cols):
    sns.scatterplot(x=df[col],y=df['Exam_Score'], ax=axes[i])

# Remove any empty subplots (if total plots < n_rows*n_cols)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.show()

# ==============================================================================================================

# Select categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()

# Automatically calculate rows and columns
n_cols = 3  # number of plots per row
n_rows = math.ceil(len(categorical_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()  # flatten in case of multiple rows

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=df, ax=axes[i], hue=col, palette="Set2")
    axes[i].set_title(f"Count Plot of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")   # countplot = frequency, not "Exam Score"
    axes[i].tick_params(axis="x")  # rotate labels if too long

# Remove any empty subplots (if total plots < n_rows*n_cols)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ==============================================================================================================

# Correlation Analysis:** Use a heatmap to detect correlations between features and the target variable.
# Compute correlations only for numeric features.
# This helps us understand how strongly each factor relates to the exam scores and to each other
corr_matrix = df.corr(numeric_only=True)

# Plot heatmap for better visualization
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.title("Correlation Heatmap of Student Performance Factors", fontsize=14)
plt.show()

# ==============================================================================================================

# Handling numerical features based on correlations
df.drop(['Sleep_Hours','Physical_Activity'],axis=1,inplace=True)


print('==============================================================================================================')

# Handling Categorical Features based on hypothesis testing
results = []
categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

for col in categorical_features:
    groups = df[col].unique()
    if len(groups) < 2:
        continue  # Skip features with only one category

    # Collect target values for each group
    group_values = [df[df[col] == g]['Exam_Score'] for g in groups]

    # Apply t-test if binary, ANOVA if more than 2 groups
    if len(groups) == 2:
        stat, p = ttest_ind(group_values[0], group_values[1], equal_var=False)
        test_used = "t-test (Welch)"
    else:
        stat, p = f_oneway(*group_values)
        test_used = "ANOVA"

    results.append({
        "feature": col,
        "test": test_used,
        "statistic": stat,
        "p_value": p,
        "state": "Has Effect" if p < 0.05 else "No Effect"
    })


hypothesis_results = pd.DataFrame(results).sort_values('p_value',ascending=False)

print(f"Hypothesis Testing\n {hypothesis_results[['feature','p_value','state']].head()}")

# dropping no effect categorical features
df.drop(hypothesis_results[hypothesis_results['state'] == 'No Effect']['feature'],axis=1,inplace=True)

print('==============================================================================================================')

# Encoding categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Scaling
scaler = StandardScaler()
for col in ['Hours_Studied','Attendance', 'Previous_Scores', 'Tutoring_Sessions']:
    df_encoded[col] = scaler.fit_transform(df_encoded[[col]])


# Model Building
# Features & target
X = df_encoded.drop(columns=["Exam_Score"])
y = df_encoded["Exam_Score"]
y_log = np.log(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log , test_size=0.2, random_state=42
)

# Removing outliers from Training data
# Select only numeric columns (int or float)
numeric_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions']

for col in numeric_cols:
    Q1 = X_train[col].quantile(0.25)        # First quartile (25%)
    Q3 = X_train[col].quantile(0.75)        # Third quartile (75%)
    IQR = Q3 - Q1                           # Interquartile Range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Check if any outliers exist in this column
    if ((X_train[col] < lower_bound) | (X_train[col] > upper_bound)).any():
        # Clip outliers into the valid range
        X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)

# y-train
Q1 = y_train.quantile(0.25)        # First quartile (25%)
Q3 = y_train.quantile(0.75)        # Third quartile (75%)
IQR = Q3 - Q1                      # Interquartile Range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Check if any outliers exist in this column
if ((y_train < lower_bound) | (y_train > upper_bound)).any():
    # Clip outliers into the valid range
    y_train = y_train.clip(lower=lower_bound, upper=upper_bound)


# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_train_pred = linear_model.predict(X_train)

print("---- Linear Regression ----")
print("Train R²:", r2_score(y_train, y_train_pred))

print('==============================================================================================================')

# Polynomial Regression Model
poly_model = PolynomialFeatures(degree=2, include_bias=False)

X_poly_train = poly_model.fit_transform(X_train)

X_poly_test = poly_model.transform(X_test)

poly_reg = LinearRegression()

poly_reg.fit(X_poly_train, y_train)

poly_reg.fit(X_poly_train,y_train)

pred = poly_reg.predict(X_poly_train)
print("---- Polynomial Regression ----")
print("Train R²:", r2_score(y_train, pred))

print('==============================================================================================================')

# Evaluation

# cross validation
scores_lin = cross_val_score(linear_model, X_train, y_train, cv=5, scoring="r2")

scores_poly = cross_val_score(
    Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
              ("model", LinearRegression())]),
    X_train, y_train, cv=5, scoring="r2"
)
print("cross validation results")
print("Linear CV R² mean:", np.mean(scores_lin), "±", np.std(scores_lin))
print("Polynomial CV R² mean:", np.mean(scores_poly), "±", np.std(scores_poly))

print('==============================================================================================================')

# Testing Accuracy
print("Testing Accuracy")
y_pred_lin = linear_model.predict(X_test)
print("---- Linear Regression ----")
print("Test R² :", r2_score(y_test,y_pred_lin))
print("RMSE    :", mean_squared_error(y_test, y_pred_lin))
print("MAE     :", mean_absolute_error(y_test, y_pred_lin))

print('\n')

y_pred_poly = poly_reg.predict(X_poly_test)
print("---- Polynomial Regression ----")
print("Test R² :",r2_score(y_test, y_pred_poly))
print("RMSE    :", mean_squared_error(y_test, y_pred_poly))
print("MAE     :", mean_absolute_error(y_test, y_pred_poly))

# ==============================================================================================================

# Plot predicted vs actual scores** to check Linear Regression model fit.

plt.figure(figsize=(8,8))

# Scatter plot of actual vs predicted scores
plt.scatter(np.exp(y_test), np.exp(y_pred_lin), alpha=0.7, color="blue", label="Predictions")

# Perfect fit reference line (y = x)
plt.plot(
    [np.exp(y_test.min()), np.exp(y_test.max())],
    [np.exp(y_test.min()), np.exp(y_test.max())],
    color="red", linestyle="--", label="Perfect Fit"
)

# Labels and title
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Linear Regression: Predicted vs Actual")

# Legend and show
plt.legend()
plt.show()

#==============================================================================================================

### Plot **predicted vs actual scores** to check **Polynomial Regression** model fit.
plt.figure(figsize=(8,8))

# Scatter plot in original scale
plt.scatter(np.exp(y_test), np.exp(y_pred_poly), alpha=0.7, color="green", label="Predictions")

# Perfect fit line in original scale
plt.plot(
    [np.exp(y_test.min()), np.exp(y_test.max())],
    [np.exp(y_test.min()), np.exp(y_test.max())],
    color="red", linestyle="--", label="Perfect Fit"
)

plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Polynomial Regression: Predicted vs Actual")
plt.legend()
plt.show()


#==============================================================================================================

# Residuals plot
residuals = np.exp(y_test) -  np.exp(y_pred_lin)
plt.figure(figsize=(6,4))
plt.scatter(np.exp(y_pred_lin), residuals, alpha=0.6, color="purple")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals Plot")
plt.show()

#==============================================================================================================