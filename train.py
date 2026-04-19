import pandas as pd
import logging
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO)

def train_model(df):
    logging.info("ML Training Started")

    # Split data
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ---------------------------
    # Model 1: Logistic Regression
    model1 = LogisticRegression(max_iter=1000)
    model1.fit(X_train, y_train)

    y_pred1 = model1.predict(X_test)

    # ---------------------------
    # Model 2: Random Forest
    model2 = RandomForestClassifier()
    model2.fit(X_train, y_train)

    y_pred2 = model2.predict(X_test)

    # ---------------------------
    # Evaluation Function
    def evaluate(y_test, y_pred):
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

    metrics1 = evaluate(y_test, y_pred1)
    metrics2 = evaluate(y_test, y_pred2)

    # ---------------------------
    # Save Metrics
    all_metrics = {
        "LogisticRegression": metrics1,
        "RandomForest": metrics2
    }

    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    # ---------------------------
    # Save Best Model (based on accuracy)
    if metrics1["accuracy"] > metrics2["accuracy"]:
        joblib.dump(model1, "model.pkl")
        best_model = "LogisticRegression"
    else:
        joblib.dump(model2, "model.pkl")
        best_model = "RandomForest"

    logging.info(f"Best model: {best_model}")
    logging.info("ML Training Completed")

    return all_metrics
