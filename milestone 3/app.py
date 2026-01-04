import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ Page Config ------------------
st.set_page_config(page_title="Study Tracker", page_icon="📊", layout="wide")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    with open("D:\Study Tracker\marks_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ------------------ Recommendation Logic ------------------
def generate_recommendations(marks, sleep, attention, exercise):
    recs = []

    if marks < 50:
        recs.append("📚 Academic performance is low. Focus on fundamentals and increase study hours.")
    elif marks < 70:
        recs.append("📘 Performance is average. Improve consistency and reduce distractions.")
    elif marks < 85:
        recs.append("📗 Good performance. Practice advanced questions.")
    else:
        recs.append("🏆 Excellent performance! Explore advanced topics.")

    if sleep < 6:
        recs.append("😴 Increase sleep to 7–8 hours for better focus.")
    elif sleep > 8:
        recs.append("⏰ Avoid oversleeping and balance your routine.")
    else:
        recs.append("✅ Sleep duration is optimal.")

    if attention == 0:
        recs.append("📵 Low attention detected. Reduce social media and use Pomodoro technique.")
    elif attention == 1:
        recs.append("🎧 Attention is average. Study in a distraction-free environment.")
    else:
        recs.append("🎯 High attention level. Maintain your habits.")

    if exercise < 0.5:
        recs.append("🏃 Add at least 30 minutes of daily physical activity.")
    elif exercise < 1:
        recs.append("💪 Slightly increase exercise to boost concentration.")
    else:
        recs.append("🔥 Excellent exercise routine.")

    return recs
#------------------- ROLE_PAGES ---------------
ROLE_PAGES = {
    "student": [
        "🏠 Home",
        "👤 Single Prediction",
        "🚪 Logout",
    ],
    "teacher": [
        "🏠 Home",
        "🧠 Train Model",
        "👤 Single Prediction",
        "📂 Batch Prediction",
        "📊 Insights",
        "🚪 Logout",
    ],
    "admin": [
        "🏠 Home",
        "🧠 Train Model",
        "👤 Single Prediction",
        "📂 Batch Prediction",
        "📊 Insights",
        "🚪 Logout",
    ],
}

def require_role(allowed_roles):
    if st.session_state.get("role") not in allowed_roles:
        st.error("❌ You do not have permission to access this page.")
        st.stop()

# ------------------ Sidebar Navigation ------------------
st.sidebar.title("📚 Study Tracker")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    role = st.session_state["role"]
    allowed_pages = ROLE_PAGES.get(role, [])

    page = st.sidebar.radio(
        "Navigation",
        allowed_pages
    )
else:
    page = "🔐 Login"

# ------------------ Database & Auth Setup ------------------
import sqlite3
import hashlib

def get_db():
    return sqlite3.connect("users.db", check_same_thread=False)

conn = get_db()
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """
)
conn.commit()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate(username, password):
    cursor.execute(
        "SELECT role FROM users WHERE username=? AND password=?",
        (username, hash_password(password)),
    )
    return cursor.fetchone()


# ------------------ Login / Signup Page ------------------
if page == "🔐 Login":
    st.title("🔐 Authentication")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            result = authenticate(username, password)
            if result:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["role"] = result[0]
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        with st.form("signup_form"):
            new_user = st.text_input("Username")
            new_pass = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["student", "teacher", "admin"])
            create = st.form_submit_button("Create Account")

        if create:
            try:
                cursor.execute(
                    "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                    (new_user, hash_password(new_pass), role),
                )
                conn.commit()
                st.success("Account created. You can now login.")
            except sqlite3.IntegrityError:
                st.error("Username already exists")

# ------------------ Logout ------------------
elif page == "🚪 Logout":
    st.session_state.clear()
    st.success("Logged out successfully")
    st.rerun()

# ------------------ Home Page ------------------
elif page == "🏠 Home":
    require_role(["student", "teacher", "admin"])
    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(135deg, #1d2671, #c33764);
            padding: 50px;
            border-radius: 24px;
            color: white;
            text-align: center;
        }
        .hero h1 {
            font-size: 42px;
            margin-bottom: 10px;
        }
        .hero p {
            font-size: 18px;
            opacity: 0.9;
        }
        .feature {
            background: #ffffff;
            padding: 22px;
            border-radius: 18px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            height: 100%;
        }
        .stat {
            background: #f3f4ff;
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero">
            <h1>📚 Study Tracker</h1>
            <p>Predict • Analyze • Improve student academic performance using data-driven insights</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="feature">
                <h3>👤 Individual Prediction</h3>
                <p>Estimate student marks and receive personalized recommendations based on habits and focus.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="feature">
                <h3>📂 Batch Analytics</h3>
                <p>Upload datasets to identify trends, top performers, and students needing intervention.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            """
            <div class="feature">
                <h3>📊 Insight Dashboard</h3>
                <p>Visualize relationships between sleep, study, attention, and academic performance.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.markdown("### 📈 What This Application Delivers")

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown("<div class='stat'>🎯 Accurate Predictions</div>", unsafe_allow_html=True)
    s2.markdown("<div class='stat'>🧠 Early Risk Detection</div>", unsafe_allow_html=True)
    s3.markdown("<div class='stat'>🚦 Performance Monitoring</div>", unsafe_allow_html=True)
    s4.markdown("<div class='stat'>📑 Actionable Insights</div>", unsafe_allow_html=True)

    st.write("")
    st.info("Use the navigation menu to begin predictions or explore analytics based on your role.")

# ------------------ Train Model Page ------------------
elif page == "🧠 Train Model":
    require_role(["teacher", "admin"])
    st.title("🧠 Train Marks Prediction Model")

    st.info("Upload a dataset to retrain the marks prediction model.")

    uploaded_file = st.file_uploader(
        "Upload Training Dataset (CSV or Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file)

        # Normalize columns
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        
        required_cols = [
            "study_hours",
            "sleep_hours",
            "social_media_hours",
            "exercise_hours",
            "attention_level",
            "test_marks",
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error("❌ Missing required columns")
            st.write(missing)
            st.stop()

        st.success("✅ Dataset validated successfully")
        st.dataframe(df.head())

        st.divider()

        # Model selection
        model_choice = st.selectbox(
            "Choose Regression Model",
            ["Ridge Regression", "Linear Regression"]
        )

        if st.button("🚀 Train Model"):
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import Ridge,LinearRegression
            from sklearn.metrics import r2_score, mean_absolute_error

            X = df[
                [
                    "study_hours",
                    "sleep_hours",
                    "social_media_hours",
                    "exercise_hours",
                    "attention_level",
                ]
            ]
            y = df["test_marks"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if model_choice == "Ridge Regression":
                model_new = Ridge(alpha=1.0)
            else:
                model_new = LinearRegression()                

            model_new.fit(X_train, y_train)
            preds = model_new.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            st.success("🎯 Model trained successfully")

            c1, c2 = st.columns(2)
            c1.metric("R² Score", f"{r2:.2f}")
            c2.metric("MAE", f"{mae:.2f}")

            # Save model
            with open("marks_prediction_model.pkl", "wb") as f:
                pickle.dump(model_new, f)

            st.success("💾 Model saved and ready for prediction")

# ------------------ Single Prediction Page ------------------
elif page == "👤 Single Prediction":
    st.title("👤 Individual Student Prediction")
    require_role(["student", "teacher", "admin"])
    with st.form("single_form"):
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.number_input("Study Hours", 0.0, 12.0, step=0.5)
            sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, step=0.5)
        with col2:
            social_hours = st.number_input("Social Media Hours", 0.0, 12.0, step=0.5)
            exercise_hours = st.number_input("Exercise Hours", 0.0, 5.0, step=0.5)

        attention = st.selectbox(
            "Attention Level",
            options=[0, 1, 2],
            format_func=lambda x: ["Distracted", "Average", "Focused"][x]
        )

        submit = st.form_submit_button("Predict Marks")

    if submit:
        features = np.array([[study_hours, sleep_hours, social_hours, exercise_hours, attention]])
        raw_pred = model.predict(features)[0]
        predicted_marks = round(min(max(raw_pred, 0), 100), 2)

        st.success(f"🎯 Predicted Marks: {predicted_marks}")

        # Insight cards instead of charts
        c1, c2, c3 = st.columns(3)
        c1.metric("Sleep Quality", "Good" if sleep_hours >= 7 else "Needs Improvement")
        c2.metric("Focus Level", ["Low", "Medium", "High"][attention])
        c3.metric("Activity Level", "Active" if exercise_hours >= 1 else "Low")

        st.subheader("📌 Personalized Recommendations")
        for r in generate_recommendations(predicted_marks, sleep_hours, attention, exercise_hours):
            st.write("•", r)

# ------------------ Batch Prediction Page ------------------
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Student Prediction")
    require_role(["teacher", "admin"])
    st.info("Upload a CSV or Excel file. Column order does not matter.")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            batch_df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            batch_df = pd.read_excel(uploaded_file)

        # Normalize column names
        batch_df.columns = (
            batch_df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        required_cols = {
            "student_name": "Student_Name",
            "study_hours": "Study_Hours",
            "sleep_hours": "Sleep_Hours",
            "social_media_hours": "Social_Media_Hours",
            "exercise_hours": "Exercise_Hours",
            "attention_level": "Attention_Level",
        }

        missing = [v for k, v in required_cols.items() if k not in batch_df.columns]
        if missing:
            st.error("❌ Missing required columns")
            st.write(missing)
            st.stop()

        batch_df = batch_df.rename(columns={k: v for k, v in required_cols.items()})
        batch_df = batch_df[list(required_cols.values())]

        features = batch_df[[
            "Study_Hours",
            "Sleep_Hours",
            "Social_Media_Hours",
            "Exercise_Hours",
            "Attention_Level"
        ]]

        batch_df["Predicted_Marks"] = model.predict(features)
        batch_df["Predicted_Marks"] = batch_df["Predicted_Marks"].apply(lambda x: round(min(max(x, 0), 100), 2))

        batch_df["Recommendations"] = batch_df.apply(
            lambda row: " | ".join(
                generate_recommendations(
                    row["Predicted_Marks"],
                    row["Sleep_Hours"],
                    row["Attention_Level"],
                    row["Exercise_Hours"],
                )
            ), axis=1)

        # Store for Insights page
        st.session_state["batch_df"] = batch_df.copy()

        st.success("✅ Batch prediction completed")
        st.dataframe(batch_df)

        st.divider()
        st.subheader("🚦 Batch Health Summary")

        avg_marks = batch_df["Predicted_Marks"].mean()
        if avg_marks >= 70:
            st.success("🟢 Healthy overall performance")
        elif avg_marks >= 50:
            st.warning("🟡 Moderate performance – needs improvement")
        else:
            st.error("🔴 Critical performance – immediate action required")

        st.divider()
        st.subheader("✨ Quick Batch Insights")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📊 Avg Marks", f"{avg_marks:.1f}")
        c2.metric("🏆 ≥75 Marks", f"{(batch_df['Predicted_Marks']>=75).mean()*100:.0f}%")
        c3.metric("😴 <6h Sleep", f"{(batch_df['Sleep_Hours']<6).mean()*100:.0f}%")
        c4.metric("⚠️ Distracted", f"{(batch_df['Attention_Level']==0).mean()*100:.0f}%")

        st.divider()
        st.subheader("🏅 Performance Extremes")

        k = max(1, int(0.05 * len(batch_df)))
        top_5 = batch_df.nlargest(k, "Predicted_Marks")
        bottom_5 = batch_df.nsmallest(k, "Predicted_Marks")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("### 🌟 Top 5% Students")
            st.dataframe(top_5[["Student_Name", "Predicted_Marks"]])

        with colB:
            st.markdown("### 🚨 Bottom 5% Students")
            st.dataframe(bottom_5[["Student_Name", "Predicted_Marks"]])

        st.divider()

        csv = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results",
            csv,
            "batch_student_predictions.csv",
            "text/csv",
        )

# ------------------ Insights Page ------------------
elif page == "📊 Insights":
    st.title("📊 Batch Insights & Relationships")
    require_role(["teacher", "admin"])
    if "batch_df" not in st.session_state:
        st.info("Run a batch prediction first to view insights.")
        st.stop()

    batch_df = st.session_state["batch_df"]

    st.subheader("📈 Feature vs Marks Relationships")

    fig1, ax1 = plt.subplots()
    ax1.scatter(batch_df["Study_Hours"], batch_df["Predicted_Marks"])
    ax1.set_xlabel("Study Hours")
    ax1.set_ylabel("Predicted Marks")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.scatter(batch_df["Sleep_Hours"], batch_df["Predicted_Marks"])
    ax2.set_xlabel("Sleep Hours")
    ax2.set_ylabel("Predicted Marks")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.scatter(batch_df["Social_Media_Hours"], batch_df["Predicted_Marks"])
    ax3.set_xlabel("Social Media Hours")
    ax3.set_ylabel("Predicted Marks")
    st.pyplot(fig3)

    st.subheader("🧠 Attention Level Impact")
    attn_avg = batch_df.groupby("Attention_Level")["Predicted_Marks"].mean()
    fig4, ax4 = plt.subplots()
    ax4.bar(["Distracted", "Average", "Focused"], attn_avg.values)
    ax4.set_ylabel("Average Predicted Marks")
    st.pyplot(fig4)

    st.subheader("🔗 Feature Correlation Overview")
    corr = batch_df[[
        "Study_Hours",
        "Sleep_Hours",
        "Social_Media_Hours",
        "Exercise_Hours",
        "Predicted_Marks"
    ]].corr()

    fig5, ax5 = plt.subplots()
    im = ax5.imshow(corr.values)
    ax5.set_xticks(range(len(corr.columns)))
    ax5.set_yticks(range(len(corr.columns)))
    ax5.set_xticklabels(corr.columns, rotation=45)
    ax5.set_yticklabels(corr.columns)
    fig5.colorbar(im)
    st.pyplot(fig5)

# ------------------ Footer ------------------
st.divider()
st.caption("Study Tracker | ML-powered Academic Analytics")
