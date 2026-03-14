import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


DATA_PATH = "data/student_dropout_dataset.csv"
MODEL_PATH = "model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"


def train_models():
    """Train classification models and persist trained artifacts."""
    df = pd.read_csv(DATA_PATH)

    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

    X = df.drop("dropout", axis=1)
    y = df["dropout"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    lr_model = LogisticRegression(max_iter=2000, random_state=42)
    lr_model.fit(X_train, y_train)

    # Evaluate
    rf_pred = rf_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)

    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
    print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

    print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))
    print("\nLogistic Regression Classification Report:\n", classification_report(y_test, lr_pred))

    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    joblib.dump(rf_model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Saved model to {MODEL_PATH} and label encoder to {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    train_models()
