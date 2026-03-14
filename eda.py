import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/student_dropout_dataset.csv"


def run_eda():
    """Run exploratory data analysis and show charts."""
    df = pd.read_csv(DATA_PATH)

    print("\n--- Head ---")
    print(df.head())

    print("\n--- Info ---")
    print(df.info())

    print("\n--- Describe ---")
    print(df.describe())

    # Missing values
    print("\n--- Missing values ---")
    print(df.isnull().sum())

    if df.isnull().any().any():
        print("Dropping rows with missing values...")
        df = df.dropna()

    # EDA plots
    plt.figure(figsize=(10, 6))
    sns.countplot(x="dropout", data=df)
    plt.title("Dropout Distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="dropout", y="attendance_percentage", data=df)
    plt.title("Attendance vs Dropout")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="dropout", y="stress_level", data=df)
    plt.title("Stress Level vs Dropout")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="study_hours_per_week",
        y="previous_grade",
        hue="dropout",
        data=df
    )
    plt.title("Study Hours vs Grade")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # 2x2 dashboard
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    sns.countplot(x="dropout", data=df, ax=ax[0, 0])
    ax[0, 0].set_title("Dropout Distribution")

    sns.boxplot(x="dropout", y="attendance_percentage", data=df, ax=ax[0, 1])
    ax[0, 1].set_title("Attendance vs Dropout")

    sns.boxplot(x="dropout", y="stress_level", data=df, ax=ax[1, 0])
    ax[1, 0].set_title("Stress vs Dropout")

    sns.scatterplot(
        x="study_hours_per_week",
        y="previous_grade",
        hue="dropout",
        data=df,
        ax=ax[1, 1]
    )
    ax[1, 1].set_title("Study Hours vs Grade")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_eda()
