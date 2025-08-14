# train_safe.py
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/heart_disease.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------
# Load data
# ------------------------
df = pd.read_csv(DATA_PATH)
print("Columns in dataset:", df.columns.tolist())

# ------------------------
# Identify target column
# ------------------------
TARGET_COL = "target" if "target" in df.columns else df.columns[-1]  # use last column if target missing
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ------------------------
# Identify categorical and numerical columns
# ------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# ------------------------
# Preprocessing
# ------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# ------------------------
# Train/Test split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Build pipeline
# ------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# ------------------------
# Train model
# ------------------------
pipeline.fit(X_train, y_train)

# ------------------------
# Save pipeline
# ------------------------
joblib.dump(pipeline, os.path.join(MODEL_DIR, "final_pipeline_safe.pkl"))
print(f"✅ Model saved to {MODEL_DIR}/final_pipeline_safe.pkl")

# ------------------------
# Evaluate
# ------------------------
score = pipeline.score(X_test, y_test)
print(f"Model accuracy on test set: {score:.4f}")
