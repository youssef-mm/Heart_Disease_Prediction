# src/hyperparameter_tuning.py
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

DATA_TRAIN = "data/heart_disease_preprocessed.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def run():
    df = pd.read_csv(DATA_TRAIN)
    X = df.drop(columns=["num"])
    y = df["num"]

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f"{MODEL_DIR}/rf_best.pkl")

    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best score: {grid_search.best_score_:.4f}")

if __name__ == "__main__":
    run()