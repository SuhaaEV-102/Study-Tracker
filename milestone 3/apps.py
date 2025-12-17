import streamlit as st
import pickle
import numpy as np
import pandas as pd
# ------------------ Page Config ------------------
st.set_page_config(page_title="Study Tracker Dashboard", page_icon="📊", layout="centered")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    with open("D:\Study Tracker\marks_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ------------------ Recommendation Logic ------------------
def generate_recommendations(marks, sleep, attention, exercise):
    recommendations = []

    # Academic Performance
    if marks < 50:
        recommendations.append("📚 Focus on fundamentals and increase daily study hours.")
    elif marks < 70:
        recommendations.append("📘 Revise regularly and reduce distractions to improve consistency.")
    elif marks < 85:
        recommendations.append("📗 Practice advanced questions to reach excellence.")
    else:
        recommendations.append("🏆 Excellent performance! Explore advanced topics.")

    # Sleep
    if sleep < 6:
        recommendations.append("😴 Increase sleep to at least 7–8 hours for better focus.")
    elif sleep > 8:
        recommendations.append("⏰ Avoid oversleeping and balance your routine.")
    else:
        recommendations.append("✅ Your sleep schedule is healthy.")

    # Attention
    if attention == 0:
        recommendations.append("📵 Reduce social media and try Pomodoro technique.")
    elif attention == 1:
        recommendations.append("🎧 Study in a distraction-free environment.")
    else:
        recommendations.append("🎯 Great focus! Maintain your habits.")

    # Exercise
    if exercise < 0.5:
        recommendations.append("🏃 Add at least 30 minutes of daily physical activity.")
    elif exercise < 1:
        recommendations.append("💪 Increase exercise slightly to boost concentration.")
    else:
        recommendations.append("🔥 Excellent exercise routine.")

    return recommendations
# ------------------ Batch Prediction Logic ------------------

import pandas as pd

def batch_predict(df, model):
    # Predict marks
    features = df[[
        "Study_Hours",
        "Sleep_Hours",
        "Social_Media_Hours",
        "Exercise_Hours",
        "Attention_Level"
    ]]
    
    df["Predicted_Marks"] = model.predict(features)
    
    # Clamp marks
    df["Predicted_Marks"] = df["Predicted_Marks"].apply(
        lambda x: round(min(max(x, 0), 100), 2)
    )

    # Generate recommendations
    df["Recommendations"] = df.apply(
        lambda row: " | ".join(
            generate_recommendations(
                row["Predicted_Marks"],
                row["Sleep_Hours"],
                row["Attention_Level"],
                row["Exercise_Hours"]
            )
        ),
        axis=1
    )

    return df

# ------------------ UI ------------------
st.title("📊 Student Study Tracker Dashboard")
st.write("Predict student marks and get personalized recommendations")

st.divider()

# ------------------ Input Form ------------------
with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input("📘 Study Hours", 0.0, 12.0, step=0.5)
        sleep_hours = st.number_input("😴 Sleep Hours", 0.0, 12.0, step=0.5)

    with col2:
        social_hours = st.number_input("📱 Social Media Hours", 0.0, 12.0, step=0.5)
        exercise_hours = st.number_input("🏃 Exercise Hours", 0.0, 5.0, step=0.5)

    attention = st.selectbox(
        "🧠 Attention Level",
        options=[0, 1, 2],
        format_func=lambda x: ["Distracted", "Average", "Focused"][x]
    )

    submit = st.form_submit_button("🎯 Predict Marks")

# ------------------ Prediction ------------------
if submit:
    features = np.array([[study_hours, sleep_hours, social_hours, exercise_hours, attention]])
    predicted_marks = round(model.predict(features)[0], 2)

    st.success(f"🎯 Predicted Marks: {predicted_marks}")

    # Performance Label
    if predicted_marks >= 85:
        st.balloons()
        st.info("🏆 Performance Level: Excellent")
    elif predicted_marks >= 70:
        st.info("📗 Performance Level: Good")
    elif predicted_marks >= 50:
        st.warning("📘 Performance Level: Average")
    else:
        st.error("📕 Performance Level: Poor")

    # Recommendations
    st.subheader("📌 Personalized Recommendations")
    recs = generate_recommendations(predicted_marks, sleep_hours, attention, exercise_hours)
    for r in recs:
        st.write("•", r)

#------------------- Batch Prediction ------------------
st.divider()
st.header("📂 Batch Student Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        batch_df = pd.read_csv(uploaded_file, encoding="latin1")
    else:
        batch_df = pd.read_excel(uploaded_file)

    required_cols = [
        "Student_Name",
        "Study_Hours",
        "Sleep_Hours",
        "Social_Media_Hours",
        "Exercise_Hours",
        "Attention_Level"
    ]

    if not all(col in batch_df.columns for col in required_cols):
        st.error("❌ Uploaded file is missing required columns")
    else:
        results_df = batch_predict(batch_df, model)

        st.success("✅ Batch prediction completed")
        st.dataframe(results_df)

        # Download results
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV",
            csv,
            "predicted_student_marks.csv",
            "text/csv"
        )

# ------------------ Footer ------------------
st.divider()
st.caption("Built with ❤️ using Streamlit | Study Tracker Project")