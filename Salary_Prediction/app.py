import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# Load Data and Model
# -----------------------------
data = pd.read_csv("data/Salary_Data.csv")
model = joblib.load("model/salary_model.pkl")

# -----------------------------
# Model Evaluation
# -----------------------------
X = data[["YearsExperience"]]
y = data["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("💰 Salary Prediction")

st.sidebar.markdown("""
### Project Information

**Machine Learning Model:**  
Linear Regression

**Dataset:**  
Salary Prediction Dataset

**Input:**  
Years of Experience

**Output:**  
Predicted Salary
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "This application predicts salary based on an employee's years of experience."
)

# -----------------------------
# Main Title
# -----------------------------
st.title("💰 Salary Prediction System")

st.markdown(
    "### Machine Learning based Salary Prediction using Linear Regression"
)

st.markdown("---")

# -----------------------------
# Project Overview
# -----------------------------
st.header("📌 Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(data))

with col2:
    st.metric("Input Feature", "Years Experience")

with col3:
    st.metric("Model", "Linear Regression")

st.markdown(
    """
This project uses **Linear Regression** to predict salary based on
the number of years of professional experience.
"""
)

# -----------------------------
# Prediction Section
# -----------------------------
st.header("💰 Predict Your Salary")

col1, col2 = st.columns([2, 1])

with col1:

    experience = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.1
    )

with col2:

    st.write("")
    st.write("")

    predict_button = st.button(
        "🔮 Predict Salary",
        use_container_width=True
    )

if predict_button:

    prediction = model.predict([[experience]])[0]

    st.success(
        f"### Predicted Salary: {prediction:,.2f}"
    )

    st.info(
        f"For **{experience:.1f} years** of experience, "
        f"the predicted salary is **{prediction:,.2f}**."
    )

# -----------------------------
# Salary Graph
# -----------------------------
st.header("📈 Experience vs Salary")

fig, ax = plt.subplots()

ax.scatter(
    data["YearsExperience"],
    data["Salary"]
)

ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.set_title("Years of Experience vs Salary")

st.pyplot(fig)

# -----------------------------
# Model Performance
# -----------------------------
st.header("🤖 Model Performance")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("MAE", f"{mae:,.2f}")

with m2:
    st.metric("MSE", f"{mse:,.2f}")

with m3:
    st.metric("RMSE", f"{rmse:,.2f}")

with m4:
    st.metric("R² Score", f"{r2:.4f}")

# -----------------------------
# Dataset Preview
# -----------------------------
st.header("📊 Dataset Preview")

st.dataframe(
    data.head(10),
    use_container_width=True
)

# -----------------------------
# Dataset Statistics
# -----------------------------
st.header("📋 Dataset Statistics")

st.dataframe(
    data.describe(),
    use_container_width=True
)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
    "🎓 **Salary Prediction ML Project | Built with Python, Scikit-learn & Streamlit**"
)