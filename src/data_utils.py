import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path: str):
    """Load the student dataset from a CSV file."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame):
    """Basic preprocessing: drop missing values and encode categorical columns."""
    df = df.dropna().reset_index(drop=True)
    if "gender" in df.columns:
        le = LabelEncoder()
        df["gender"] = le.fit_transform(df["gender"])
        return df, le
    return df, None
