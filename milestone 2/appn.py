import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# -------------------- Load & Train Models --------------------
@st.cache_data
def load_data():
    df = pd.read_excel("D:\Study Tracker\milestone 2\student_data.xlsx")

    # Generate marks if not present
    if "Test_Marks" not in df.columns:
        np.random.seed(42)
        df["Test_Marks"] = np.random.randint(0, 101, len(df))

    return df


@st.cache_resource
def train_models(df):

    # Linear Regression → Marks
    X_marks = df[["Study_Hours", "Sleep_Hours", "Social_Media_Hours", "Exercise_Hours", "Attention_Level"]]
    y_marks = df["Test_Marks"]

    lr = LinearRegression().fit(X_marks, y_marks)

    # KNN → Attention Level
    X_att = df[["Study_Hours", "Sleep_Hours", "Social_Media_Hours", "Exercise_Hours"]]
    y_att = df["Attention_Level"]

    knn = KNeighborsRegressor(n_neighbors=5).fit(X_att, y_att)

    return lr, knn


# -------------------- Recommendation Engine --------------------
def get_recommendations(marks, attention, study, sleep, social, exercise):

    rec = []

    # Marks based recommendations
    if marks < 40:
        rec.append("⚠️ Marks are low. Increase study time by 1–2 hours per day.")
        rec.append("Avoid social media during study hours.")
    elif marks < 70:
        rec.append("📘 Good! But you can improve—revise daily for 30 minutes.")
    else:
        rec.append("🎉 Excellent performance! Maintain your current routine.")

    # Attention based recommendations
    if attention < 4:
        rec.append("🧠 Low attention — try mindfulness or short study sprints.")
        rec.append("Reduce distractions: phone, noise, multitasking.")
    elif attention < 7:
        rec.append("📚 Your attention is moderate — use Pomodoro (25–5 cycles).")
    else:
        rec.append("🔥 High attention — keep practicing deep work sessions!")

    # Lifestyle recommendations
    if sleep < 6:
        rec.append("😴 Sleep less than 6 hours — increase to 7–8 hours.")
    if social > 4:
        rec.append("📵 Reduce social media usage — target < 2 hours/day.")
    if exercise < 1:
        rec.append("🏃 Add 20–30 min exercise to boost concentration.")

    return rec


# -------------------- STREAMLIT UI --------------------
def main():

    st.title("📊 Student Performance Prediction Dashboard")
    st.write("AI-powered dashboard to predict marks, attention level, and study recommendations.")

    df = load_data()
    lr, knn = train_models(df)

    st.header("🧮 Enter Student Lifestyle Details")

    col1, col2 = st.columns(2)

    with col1:
        study = st.number_input(
        "📘 Study Hours per Day",
        min_value=0.0,
        max_value=12.0,
        value=3.0,
        step=0.5,
        format="%.1f",
        help="How many hours does the student study per day?"
        )

        sleep = st.number_input(
        "😴 Sleep Hours per Day",
        min_value=0.0,
        max_value=12.0,
        value=7.0,
        step=0.5,
        format="%.1f"
        )

        exercise = st.number_input(
        "🏃 Exercise Hours per Day",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.5,
        format="%.1f"
        )

    with col2:
        social = st.number_input(
        "📱 Social Media Hours per Day",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        format="%.1f"
        )

        attention_manual = st.slider(
        "🎯 Current Attention Level (1–10)",
        min_value=1,
        max_value=10,
        value=7,
        help="Self-reported focus level"
        )

        study_goal = st.selectbox(
        "🎓 Study Goal",
        ["Improve Grades", "Maintain Performance", "Exam Preparation", "Reduce Distractions"]
        )

# Optional notes input
    notes = st.text_area("📝 Additional Notes (optional)", placeholder="Enter any special conditions...")

    st.markdown("---")

    # Data for models
    input_marks = pd.DataFrame([[study, sleep, social, exercise, attention_manual]],
                               columns=["Study_Hours", "Sleep_Hours", "Social_Media_Hours", "Exercise_Hours", "Attention_Level"])

    input_att = pd.DataFrame([[study, sleep, social, exercise]],
                             columns=["Study_Hours", "Sleep_Hours", "Social_Media_Hours", "Exercise_Hours"])

    st.subheader("🎯 Predictions")

    # Predict Marks
    predicted_marks = lr.predict(input_marks)[0]
    st.metric("Predicted Test Marks", f"{predicted_marks:.2f} / 100")

    # Predict Attention Level
    predicted_attention = knn.predict(input_att)[0]
    st.metric("Predicted Attention Level", f"{predicted_attention:.2f} / 10")

    # Recommendations
    st.subheader("📘 Personalized Recommendations")

    recs = get_recommendations(predicted_marks, predicted_attention, study, sleep, social, exercise)

    for r in recs:
        st.write("- " + r)


# Run App
if __name__ == "__main__":
    main()
