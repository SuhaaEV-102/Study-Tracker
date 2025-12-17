import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# ----------------------------- Load Data -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("D:\Study Tracker\milestone 2\student_data_with_marks.xlsx")

    # If Test_Marks is missing, generate random ones
    if "Test_Marks" not in df.columns:
        np.random.seed(42)
        df["Test_Marks"] = np.random.randint(0, 101, len(df))

    return df


# ----------------------------- Train Models -----------------------------
@st.cache_resource
def train_models(df):

    # 1) Test Marks Prediction
    X_marks = df[["Study_Hours", "Sleep_Hours", "Social_Media_Hours", "Exercise_Hours", "Attention_Level"]]
    y_marks = df["Test_Marks"]

    lr = LinearRegression().fit(X_marks, y_marks)
    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_marks, y_marks)

    # 2) Attention Level Prediction
    X_att = df[["Study_Hours", "Sleep_Hours", "Social_Media_Hours", "Exercise_Hours"]]
    y_att = df["Attention_Level"]

    dt = DecisionTreeRegressor(random_state=42).fit(X_att, y_att)
    knn = KNeighborsRegressor(n_neighbors=5).fit(X_att, y_att)

    # 3) Study Hours Prediction
    X_study = df[["Sleep_Hours", "Social_Media_Hours", "Exercise_Hours", "Attention_Level"]]
    y_study = df["Study_Hours"]

    mlr = LinearRegression().fit(X_study, y_study)

    return lr, rf, dt, knn, mlr


# ----------------------------- Streamlit UI -----------------------------
def main():

    st.title("📘 Student Performance Prediction Dashboard")
    st.write("Predict Marks, Attention, and Suggested Study Hours based on student lifestyle data.")

    df = load_data()
    lr, rf, dt, knn, mlr = train_models(df)

    st.sidebar.header("Enter Student Details")

    # User Inputs
    study = st.sidebar.number_input("Study Hours", 0, 12, 3)
    sleep = st.sidebar.number_input("Sleep Hours", 0, 12, 7)
    social = st.sidebar.number_input("Social Media Hours", 0, 12, 2)
    exercise = st.sidebar.number_input("Exercise Hours", 0, 5, 1)
    attention = st.sidebar.number_input("Attention Level (1–10)", 1, 10, 7)

    # Convert to DataFrame for model input
    input_data_marks = pd.DataFrame([[study, sleep, social, exercise, attention]],
                                    columns=["Study_Hours", "Sleep_Hours", "Social_Media_Hours", "Exercise_Hours", "Attention_Level"])

    input_data_att = pd.DataFrame([[study, sleep, social, exercise]],
                                  columns=["Study_Hours", "Sleep_Hours", "Social_Media_Hours", "Exercise_Hours"])

    input_data_study = pd.DataFrame([[sleep, social, exercise, attention]],
                                    columns=["Sleep_Hours", "Social_Media_Hours", "Exercise_Hours", "Attention_Level"])

    st.subheader("🔮 Predictions")

    # 1) Predict Test Marks
    st.write("### 📘 Predicted Test Marks")
    marks_lr = lr.predict(input_data_marks)[0]
    marks_rf = rf.predict(input_data_marks)[0]

    st.write(f"**Linear Regression:** {marks_lr:.2f} / 100")
    st.write(f"**Random Forest:** {marks_rf:.2f} / 100")

    # 2) Predict Attention Level
    st.write("### 🎯 Predicted Attention Level")
    pred_dt = dt.predict(input_data_att)[0]
    pred_knn = knn.predict(input_data_att)[0]

    st.write(f"**Decision Tree:** {pred_dt:.2f}")
    st.write(f"**KNN:** {pred_knn:.2f}")

    # 3) Predict Required Study Hours
    st.write("### ⏳ Suggested Study Hours Per Day")
    pred_study = mlr.predict(input_data_study)[0]

    st.write(f"**Recommended Study Hours:** {pred_study:.2f} hrs/day")



# Run app
if __name__ == "__main__":
    main()
