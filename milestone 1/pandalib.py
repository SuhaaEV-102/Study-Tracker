import pandas as pd

# Load dataset
df = pd.read_excel("rawdata.xlsx")

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# Remove completely empty columns
df = df.dropna(axis=1, how='all')

# Display first rows
print("Cleaned Data:")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))
