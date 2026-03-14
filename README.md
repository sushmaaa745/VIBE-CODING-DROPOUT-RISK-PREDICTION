# Student Dropout Prediction Project

This project demonstrates an end-to-end machine learning pipeline:

- **Exploratory Data Analysis (EDA)** via `eda.py` and Jupyter Notebook
- **Model training + evaluation** via `model_training.py`
- **Interactive prediction app** via Streamlit (`streamlit_app.py`)

---

## Setup

```bash
# Activate your venv (Windows PowerShell)
& ".\.venv\Scripts\Activate.ps1"

# Install dependencies
pip install -r requirements.txt
```

## Run EDA

```bash
python eda.py
```

## Train the model

```bash
python model_training.py
```

This creates `model.pkl` and `label_encoder.pkl`.

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Then open the URL shown in the terminal to interact with the app.

## Deploy to Streamlit Cloud (Live)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in.
3. Click **New app**, choose your repo, branch, and set the entry point to `streamlit_app.py`.
4. Click **Deploy** and the app will be hosted live (and auto-redeploy on updates).

## Data

The dataset is stored under `data/student_dropout_dataset.csv`.
