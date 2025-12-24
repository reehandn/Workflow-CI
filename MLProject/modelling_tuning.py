# ======================================
# IMPORT & ENV FIX
# ======================================
import os
from mlflow.tracking import MlflowClient

# FIX: Hapus parent run dari mlflow project (WAJIB UNTUK CI + DAGSHUB)
if "MLFLOW_RUN_ID" in os.environ:
    del os.environ["MLFLOW_RUN_ID"]

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ======================================
# DAGSHUB INIT (WAJIB)
# ======================================
dagshub.init(
    repo_owner="reehandn",
    repo_name="Workflow-CI",
    mlflow=True
)

# ======================================
# MLFLOW EXPERIMENT
# ======================================
mlflow.set_experiment("Adult Income Advanced Tuning")

# ======================================
# LOAD DATASET
# ======================================
df = pd.read_csv("adult_preprocessed.csv")

X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================
# HYPERPARAMETER GRID
# ======================================
param_grid = [
    {"C": 0.1, "solver": "lbfgs"},
    {"C": 1.0, "solver": "lbfgs"},
    {"C": 10.0, "solver": "lbfgs"},
]

# ======================================
# TRAINING LOOP
# ======================================
for params in param_grid:
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        print("MLflow version:", mlflow.__version__)
        print("RUN_ID (script):", run_id)
        print("Artifact URI:", mlflow.get_artifact_uri())

        # -------- Log Parameters --------
        mlflow.log_param("C", params["C"])
        mlflow.log_param("solver", params["solver"])
        mlflow.log_param("penalty", "l2")
        mlflow.log_param("max_iter", 1000)

        # -------- Train Model --------
        model = LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            penalty="l2",
            max_iter=1000
        )
        model.fit(X_train, y_train)

        # -------- Prediction --------
        y_pred = model.predict(X_test)

        # -------- Metrics --------
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # -------- Confusion Matrix --------
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png")

        # -------- Classification Report --------
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)

        mlflow.log_artifact("classification_report.txt")

        client = MlflowClient()
        print("Artifacts root (before log_model):",
            [a.path for a in client.list_artifacts(run_id, "")])

        # Log Model
        mlflow.sklearn.log_model(model, "model")

        # sesudah log_model, cek apakah folder model muncul
        print("Artifacts root (after log_model):",
            [a.path for a in client.list_artifacts(run_id, "")])
        print("Artifacts in 'model' after log_model:",
            [a.path for a in client.list_artifacts(run_id, "model")])

        print(
            f"Finished run | C={params['C']} | "
            f"accuracy={acc:.4f} | f1={f1:.4f}"
        )