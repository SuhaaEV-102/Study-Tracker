import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ----------------------------- Load Dataset -----------------------------
def load_data():
    df = pd.read_excel("D:\Study Tracker\milestone 2\student_data.xlsx")
    
    # If Test_Marks doesn't exist, generate random marks
    if "Test_Marks" not in df.columns:
        np.random.seed(42)
        df["Test_Marks"] = np.random.randint(0, 101, len(df))
    
    return df


# ---------------------- 1) Predict Test Marks ---------------------------
def predict_test_marks(df):
    print("\n=== TEST MARKS PREDICTION ===")

    X = df[["Study_Hours", "Sleep_Hours", "Social_Media_Hours", "Exercise_Hours", "Attention_Level"]]
    y = df["Test_Marks"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    print("\nLinear Regression:")
    print("MSE:", mean_squared_error(y_test, lr_pred))
    print("R2 Score:", r2_score(y_test, lr_pred))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print("\nRandom Forest:")
    print("MSE:", mean_squared_error(y_test, rf_pred))
    print("R2 Score:", r2_score(y_test, rf_pred))


# ---------------------- 2) Predict Attention Level ----------------------
def predict_attention_level(df):
    print("\n=== ATTENTION LEVEL PREDICTION ===")

    X = df[["Study_Hours", "Sleep_Hours", "Social_Media_Hours", "Exercise_Hours"]]
    y = df["Attention_Level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    print("\nDecision Tree:")
    print("MAE:", mean_absolute_error(y_test, dt_pred))

    # KNN
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    print("\nKNN:")
    print("MAE:", mean_absolute_error(y_test, knn_pred))


# ---------------------- 3) Predict Study Hours Needed -------------------
def predict_study_hours(df):
    print("\n=== STUDY HOURS PREDICTION ===")

    X = df[["Sleep_Hours", "Social_Media_Hours", "Exercise_Hours", "Attention_Level"]]
    y = df["Study_Hours"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Multiple Linear Regression
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    mlr_pred = mlr.predict(X_test)

    print("\nMultiple Linear Regression:")
    print("MSE:", mean_squared_error(y_test, mlr_pred))
    print("R2 Score:", r2_score(y_test, mlr_pred))


# ----------------------------- Main Function ----------------------------
def main():
    df = load_data()
    
    predict_test_marks(df)
    predict_attention_level(df)
    predict_study_hours(df)

# Run everything
main()
