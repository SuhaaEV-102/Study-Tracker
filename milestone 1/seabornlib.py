import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("rawdata.xlsx")
df = df.loc[:, ~df.columns.duplicated()]
df = df.dropna(axis=1, how='all')

# Pairplot

# Study Hours vs Attention Level
sns.scatterplot(data=df, x="Study Hours/Day", y="Attention Level (1-10)")
plt.title("Study Hours vs Attention Level")
plt.show()

# Distribution of Sleep Hours
sns.histplot(df["Sleep Hours"], kde=True)
plt.title("Distribution of Sleep Hours")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
