import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Argparse untuk parameter training ---
parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", type=int, default=1000)
args = parser.parse_args()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "adult_preprocessed.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

# --- Load dataset ---
df = pd.read_csv(DATA_PATH)
X = df.drop("income", axis=1)
y = df["income"]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

input_example = X_train.iloc[:5]

# --- MLflow training ---
mlflow.set_experiment("Adult Income Advance Tuning")

with mlflow.start_run() as run:
    model = LogisticRegression(max_iter=args.max_iter)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # --- Log parameters & metrics ---
    mlflow.log_param("max_iter", args.max_iter)
    mlflow.log_metric("accuracy", acc)

    # --- Log model ---
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=input_example
    )

    # --- Save local artifacts ---
    joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
    run_id = run.info.run_id
    with open(os.path.join(BASE_DIR, "run_id.txt"), "w") as f:
        f.write(run_id)

    print(f"Accuracy: {acc}")
    print(f"Run ID: {run_id}")

print("Training selesai")
