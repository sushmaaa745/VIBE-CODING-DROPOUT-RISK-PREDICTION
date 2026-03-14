import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from src.data_utils import load_data, preprocess

st.set_page_config(
    page_title="Vibe Coding - Student Dropout Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# Paths
DATA_PATH = cfg["data_path"]
MODEL_PATH = cfg["model_path"]
LABEL_ENCODER_PATH = cfg["label_encoder_path"]


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = load_data(DATA_PATH)
    df, _ = preprocess(df)
    return df


@st.cache_resource(show_spinner=False)
def load_model() -> tuple:
    """Load trained model artifacts."""
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return model, le


@st.cache_data(show_spinner=False)
def compute_metrics(df: pd.DataFrame, _model):
    """Compute evaluation metrics on a train/test split."""
    le = LabelEncoder()
    df_temp = df.copy()
    df_temp["gender"] = le.fit_transform(df_temp["gender"])

    X = df_temp.drop("dropout", axis=1)
    y = df_temp["dropout"]

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pred = _model.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)
    cm = confusion_matrix(y_test, pred)

    return report, cm


def plot_correlation(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation")
    return fig


def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return None

    importance = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    df_imp = df_imp.sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_imp, x="importance", y="feature", ax=ax, palette="viridis")
    ax.set_title("Feature Importance")
    return fig


def build_prediction_input(df: pd.DataFrame):
    st.sidebar.header("Student Profile")

    age = st.sidebar.slider("Age", int(df["age"].min()), int(df["age"].max()), int(df["age"].median()))
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    attendance = st.sidebar.slider(
        "Attendance %",
        int(df["attendance_percentage"].min()),
        int(df["attendance_percentage"].max()),
        int(df["attendance_percentage"].median()),
    )
    study_hours = st.sidebar.slider(
        "Study Hours/Week",
        int(df["study_hours_per_week"].min()),
        int(df["study_hours_per_week"].max()),
        int(df["study_hours_per_week"].median()),
    )
    grade = st.sidebar.slider(
        "Previous Grade",
        int(df["previous_grade"].min()),
        int(df["previous_grade"].max()),
        int(df["previous_grade"].median()),
    )
    stress = st.sidebar.slider(
        "Stress Level",
        int(df["stress_level"].min()),
        int(df["stress_level"].max()),
        int(df["stress_level"].median()),
    )
    sleep = st.sidebar.slider(
        "Sleep Hours",
        int(df["sleep_hours"].min()),
        int(df["sleep_hours"].max()),
        int(df["sleep_hours"].median()),
    )
    income = st.sidebar.number_input(
        "Family Income",
        min_value=int(df["family_income"].min()),
        max_value=int(df["family_income"].max()),
        value=int(df["family_income"].median()),
    )

    return {
        "age": age,
        "gender": gender,
        "attendance_percentage": attendance,
        "study_hours_per_week": study_hours,
        "previous_grade": grade,
        "stress_level": stress,
        "sleep_hours": sleep,
        "family_income": income,
    }


def main():
    st.title("🎯 Vibe Coding — Student Dropout Predictor")
    st.write(
        "This app lets you explore the dataset, visualize patterns, and predict dropout risk in real time."
    )

    df = load_dataset()
    model, le = load_model()

    tabs = st.tabs(["Dashboard", "Predict", "Model"])

    with tabs[0]:
        st.header("📊 Dashboard")
        st.markdown("### Data preview")
        st.dataframe(df.head(10))

        st.markdown("### Distribution of dropout (0=stay, 1=drop")
        fig = plt.figure(figsize=(6, 4))
        sns.countplot(x="dropout", data=df)
        plt.title("Dropout distribution")
        st.pyplot(fig)

        st.markdown("### Correlation heatmap")
        st.pyplot(plot_correlation(df))

        st.markdown("### Numeric feature distributions")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        fig, axes = plt.subplots(len(numeric_cols) // 3 + 1, 3, figsize=(18, 4 * (len(numeric_cols) // 3 + 1)))
        axes = axes.flatten()
        for ax, col in zip(axes, numeric_cols):
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)
        for ax in axes[len(numeric_cols) :]:
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

    with tabs[1]:
        st.header("🧠 Predict Dropout Risk")
        inputs = build_prediction_input(df)

        gender_encoded = le.transform([inputs["gender"]])[0]
        X = np.array([
            inputs["age"],
            gender_encoded,
            inputs["attendance_percentage"],
            inputs["study_hours_per_week"],
            inputs["previous_grade"],
            inputs["stress_level"],
            inputs["sleep_hours"],
            inputs["family_income"],
        ]).reshape(1, -1)

        prob = model.predict_proba(X)[0]
        score = float(prob[1])

        st.metric(label="Dropout Risk Score", value=f"{score:.2f}")

        if st.button("Predict now"):
            pred = model.predict(X)[0]
            if pred == 1:
                st.error("High dropout risk — consider additional support and monitoring.")
            else:
                st.success("Low dropout risk — continue current support plan.")

        st.markdown("---")
        st.markdown("**Input set**")
        st.json(inputs)

    with tabs[2]:
        st.header("🧮 Model Insights")
        report, cm = compute_metrics(df, model)

        st.markdown("### Classification report")
        st.write(pd.DataFrame(report).T)

        st.markdown("### Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("### Feature importance")
        fi = plot_feature_importance(model, df.drop("dropout", axis=1).columns.tolist())
        if fi is not None:
            st.pyplot(fi)
        else:
            st.info("Feature importance is not available for this model type.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ for Vibe Coding")


if __name__ == "__main__":
    main()
