import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_excel("D:\Study Tracker\milestone 3\student_data_with_marks.xlsx")

# Generate marks
np.random.seed(42)
df["Marks"] = (
    df["Study_Hours"] * 8
    + df["Sleep_Hours"] * 3
    - df["Social_Media_Hours"] * 5
    + df["Exercise_Hours"] * 4
    + df["Attention_Level"] * 10
    + np.random.normal(0, 5, len(df))
).clip(0, 100)

# Features & target
X = df[[
    "Study_Hours",
    "Sleep_Hours",
    "Social_Media_Hours",
    "Exercise_Hours",
    "Attention_Level"
]]
y = df["Marks"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

df.to_excel('D:/Study Tracker/milestone 3/student_clusters.xlsx', index=False)
# Save model
with open("marks_prediction_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Random Forest model saved")
