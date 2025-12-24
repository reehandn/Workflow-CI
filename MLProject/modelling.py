import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")

df = pd.read_csv("adult_preprocessed.csv")

X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model"
)

print("Accuracy:", acc)
