
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

MODEL_FILE = "diabetes_model.pkl"

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    data = pd.read_csv(url, header=None, names=columns)

    # Replace 0s in key columns with NaNs
    na_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    data[na_cols] = data[na_cols].replace(0, np.nan)
    data.fillna(data.median(numeric_only=True), inplace=True)

    return data

def train_and_save_model(model_type='Logistic Regression'):
    data = load_data()
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    if model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    pipeline.fit(X, y)
    joblib.dump((pipeline, X.columns), MODEL_FILE)
    return pipeline, X.columns

def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_and_save_model()
    return joblib.load(MODEL_FILE)

def predict_and_display(pipeline, features, input_data):
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result")
    if probability >= 0.75:
        st.error(f"⚠️ High Risk of Diabetes ({probability*100:.2f}%)")
    elif probability >= 0.5:
        st.warning(f"⚠️ Moderate Risk of Diabetes ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({probability*100:.2f}%)")

    st.metric("Prediction", f"{'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    return probability

def show_feature_importance(model, features):
    if hasattr(model.named_steps['classifier'], 'coef_'):
        coef = model.named_steps['classifier'].coef_[0]
    elif hasattr(model.named_steps['classifier'], 'feature_importances_'):
        coef = model.named_steps['classifier'].feature_importances_
    else:
        return

    st.subheader("📊 Feature Importance")
    df = pd.DataFrame({'Feature': features, 'Importance': coef})
    df["Abs"] = np.abs(df["Importance"])
    df = df.sort_values("Abs", ascending=False).drop("Abs", axis=1)
    st.dataframe(df.style.background_gradient(cmap='coolwarm'))

def main():
    st.title("🩺 Advanced Diabetes Risk Checker")
    model_type = st.selectbox("Choose AI Model", ["Logistic Regression", "Random Forest"])
    pipeline, features = train_and_save_model(model_type)

    st.sidebar.header("Enter Patient Information")
    pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose Level", 40, 200, 100)
    blood_pressure = st.sidebar.slider("Blood Pressure", 30, 130, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin Level", 0, 900, 80)
    bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 10, 100, 30)

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, dpf, age]])
    probability = predict_and_display(pipeline, features, input_data)

    st.subheader("🧾 Patient Summary")
    st.json({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    })

    show_feature_importance(pipeline, features)

    st.subheader("📥 Batch Prediction")
    uploaded = st.file_uploader("Upload CSV (with same column names)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        predictions = pipeline.predict(df)
        probs = pipeline.predict_proba(df)[:,1]
        df['Prediction'] = predictions
        df['Probability'] = probs
        st.write(df)
        st.download_button("Download Results", df.to_csv(index=False), "predictions.csv")

    st.subheader("📈 Probability Distribution")
    data = load_data()
    proba = pipeline.predict_proba(data.drop("Outcome", axis=1))[:,1]
    sns.histplot(proba, bins=20, kde=True)
    plt.axvline(probability, color='red', linestyle='--')
    plt.xlabel("Predicted Probability of Diabetes")
    plt.title("Distribution of Predicted Diabetes Risk")
    st.pyplot(plt.gcf())

if __name__ == "__main__":
    main()
