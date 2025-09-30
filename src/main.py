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
