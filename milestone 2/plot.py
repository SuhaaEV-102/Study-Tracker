import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_excel("./milestone 2/student_clusters.xlsx")

# Generate random whole-number marks between 0 and 100
np.random.seed(42)   # for reproducible results
df["Test_Marks"] = np.random.randint(0, 101, size=len(df))

# Preview
print(df.head())

# Save updated dataset (optional)
df.to_excel("./milestone 2/student_data_with_marks.xlsx", index=False)
