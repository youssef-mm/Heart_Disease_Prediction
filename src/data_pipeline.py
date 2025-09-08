import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# ------------------------
# 1. تحميل البيانات
# ------------------------
df = pd.read_csv("data/heart_disease.csv")  # غير الاسم لو الملف مختلف

# التأكد من عدم وجود قيم مفقودة
df = df.dropna()

# تحديد العمود الهدف
target = "num"
X = df.drop(columns=[target])
y = df[target]

# ------------------------
# 2. تحديد الأعمدة
# ------------------------
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# ------------------------
# 3. إعداد المعالجة المسبقة
# ------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ------------------------
# 4. إنشاء الـ Pipeline
# ------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# ------------------------
# 5. تدريب الموديل
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# ------------------------
# 6. حفظ الموديل
# ------------------------
joblib.dump(pipeline, "models/final_pipeline.pkl")

print("✅ تم تدريب وحفظ الـ pipeline بنجاح!")
