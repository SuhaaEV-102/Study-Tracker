import numpy as np
import pandas as pd

df = pd.read_excel("rawdata.xlsx")
df = df.loc[:, ~df.columns.duplicated()]
df = df.dropna(axis=1, how='all')

# Convert relevant columns to NumPy arrays
study = df["Study Hours/Day"].to_numpy()
sleep = df["Sleep Hours"].to_numpy()
attention = df["Attention Level (1-10)"].to_numpy()

# Basic statistics
print("Mean Study Hours:", np.mean(study))
print("Median Study Hours:", np.median(study))
print("Std Dev Study Hours:", np.std(study))

print("\nMean Attention Level:", np.mean(attention))
print("Variance in Attention Level:", np.var(attention))

# Correlation using NumPy
correlation = np.corrcoef(study, attention)
print("\nCorrelation between Study Hours & Attention Level:")
print(correlation)

# Outlier detection using Z-score
z_scores = (study - np.mean(study)) / np.std(study)
outliers = np.where(np.abs(z_scores) > 2)
print("\nOutliers in Study Hours (index positions):", outliers)
