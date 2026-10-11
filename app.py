import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Suppress minor compatibility warnings for clean UI
warnings.filterwarnings("ignore")

# ----------------------------------------------------
# Configuration and Setup
# ----------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Determine base paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_MODEL_DIRS = [
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "..", "models"),
    os.path.join(BASE_DIR, ".pkl"),
    os.path.join(BASE_DIR, "..", ".pkl"),
]

def find_file(filename):
    """Search for a file across possible model directories."""
    for d in POSSIBLE_MODEL_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

# ----------------------------------------------------
# Load ML Components (Cached for Performance)
# ----------------------------------------------------
@st.cache_resource
def load_ml_assets():
    """Load preprocessor, PCA, expected columns, and all available models."""
    try:
        # Load columns
        cols_path = find_file("columns.pkl")
        if not cols_path:
            raise FileNotFoundError("columns.pkl not found in models/ or .pkl/")
        expected_cols = joblib.load(cols_path)
        if "num" in expected_cols:
            expected_cols = [c for c in expected_cols if c != "num"]

        # Load Scaler
        scaler_path = find_file("scaler.pkl")
        if not scaler_path:
            raise FileNotFoundError("scaler.pkl not found in models/ or .pkl/")
        scaler = joblib.load(scaler_path)

        # Load PCA
        pca_path = find_file("pca.pkl")
        if not pca_path:
            raise FileNotFoundError("pca.pkl not found in models/ or .pkl/")
        pca = joblib.load(pca_path)

        # Load Available Models
        model_files = {
            "XGBoost (Recommended)": "final_model_XGBoost_model.pkl",
            "Voting Classifier (Ensemble)": "Voting_model.pkl",
            "Random Forest": "RF_model.pkl",
            "Support Vector Machine (SVM)": "SVM_model.pkl",
            "Decision Tree": "DT_model.pkl",
            "Logistic Regression": "log.pkl"
        }

        models = {}
        for display_name, fname in model_files.items():
            m_path = find_file(fname)
            if m_path and os.path.exists(m_path):
                try:
                    models[display_name] = joblib.load(m_path)
                except Exception as e:
                    pass

        if not models:
            raise RuntimeError("No trained model files found in models/ directory!")

        return expected_cols, scaler, pca, models

    except Exception as e:
        st.error(f"❌ Error loading model assets: {e}")
        st.stop()

expected_cols, scaler, pca, available_models = load_ml_assets()

# ----------------------------------------------------
# Sidebar: Model Selection & Information
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=70)
    st.title("Settings & Info")
    
    selected_model_name = st.selectbox(
        "🤖 Select Prediction Model",
        options=list(available_models.keys()),
        index=0
    )
    active_model = available_models[selected_model_name]
    
    st.markdown("---")
    st.subheader("💡 Presets for Quick Testing")
    preset = st.radio(
        "Load sample data:",
        ["None (Custom)", "Low Risk Example", "High Risk Example"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About This App
    This diagnostic tool predicts **cardiovascular disease risk** using machine learning trained on the clinical UCI Heart Disease dataset.
    
    - **Features**: 13 patient indicators
    - **Preprocessing**: Robust scaling & one-hot encoding
    - **Dimensionality Reduction**: Principal Component Analysis (PCA)
    """)

# ----------------------------------------------------
# Preset Defaults
# ----------------------------------------------------
default_vals = {
    'age': 50, 'sex': 'Male', 'cp': 'typical angina',
    'trestbps': 125, 'chol': 210, 'fbs': 'False',
    'restecg': 'normal', 'thalch': 155, 'exang': 'False',
    'oldpeak': 0.8, 'slope': 'upsloping', 'ca': 0, 'thal': 'normal'
}

if preset == "Low Risk Example":
    default_vals = {
        'age': 32, 'sex': 'Female', 'cp': 'non-anginal',
        'trestbps': 115, 'chol': 180, 'fbs': 'False',
        'restecg': 'normal', 'thalch': 175, 'exang': 'False',
        'oldpeak': 0.0, 'slope': 'upsloping', 'ca': 0, 'thal': 'normal'
    }
elif preset == "High Risk Example":
    default_vals = {
        'age': 65, 'sex': 'Male', 'cp': 'asymptomatic',
        'trestbps': 160, 'chol': 295, 'fbs': 'True',
        'restecg': 'lv hypertrophy', 'thalch': 110, 'exang': 'True',
        'oldpeak': 3.2, 'slope': 'downsloping', 'ca': 2, 'thal': 'reversable defect'
    }

# ----------------------------------------------------
# Main Page Header
# ----------------------------------------------------
st.title("❤️ Heart Disease Risk Prediction System")
st.markdown("""
Enter the patient's clinical markers and vitals below to evaluate their cardiovascular risk profile.
""")

st.warning("⚠️ **Medical Disclaimer**: This system is designed for educational and decision-support purposes only. It does not replace formal clinical diagnosis by a licensed physician.", icon="⚕️")

# ----------------------------------------------------
# Input Form
# ----------------------------------------------------
st.markdown("### 📋 Patient Clinical Profile")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Demographics & Basic Vitals")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=int(default_vals['age']), step=1)
    
    sex_options = ["Male", "Female"]
    sex = st.selectbox("Biological Sex", sex_options, index=sex_options.index(default_vals['sex']))
    
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=int(default_vals['trestbps']), step=1)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=int(default_vals['chol']), step=1)
    thalch = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=50, max_value=250, value=int(default_vals['thalch']), step=1)
    
    fbs_map = {
        "Normal (<= 120 mg/dl)": "False",
        "High (> 120 mg/dl)": "True"
    }
    fbs_inv_map = {v: k for k, v in fbs_map.items()}
    fbs_label = st.selectbox("Fasting Blood Sugar", list(fbs_map.keys()), index=0 if default_vals['fbs'] == 'False' else 1)
    fbs = fbs_map[fbs_label]

with col2:
    st.markdown("#### 🫀 Electrocardiography & Cardiac Tests")
    
    cp_map = {
        "Typical Angina (Exertional chest discomfort)": "typical angina",
        "Atypical Angina (Non-exertional chest pain)": "atypical angina",
        "Non-anginal Pain (Discomfort without coronary cause)": "non-anginal",
        "Asymptomatic (No pain symptoms)": "asymptomatic"
    }
    cp_default_label = [k for k, v in cp_map.items() if v == default_vals['cp']][0]
    cp_label = st.selectbox("Chest Pain Type", list(cp_map.keys()), index=list(cp_map.keys()).index(cp_default_label))
    cp = cp_map[cp_label]

    restecg_map = {
        "Normal": "normal",
        "ST-T Wave Abnormality (Inversions / elevation)": "st-t abnormality",
        "Left Ventricular Hypertrophy (LVH)": "lv hypertrophy"
    }
    restecg_default_label = [k for k, v in restecg_map.items() if v == default_vals['restecg']][0]
    restecg_label = st.selectbox("Resting ECG Results", list(restecg_map.keys()), index=list(restecg_map.keys()).index(restecg_default_label))
    restecg = restecg_map[restecg_label]

    exang_map = {
        "No (Absence of exercise-induced angina)": "False",
        "Yes (Exercise induces angina)": "True"
    }
    exang_label = st.selectbox("Exercise-Induced Angina", list(exang_map.keys()), index=0 if default_vals['exang'] == 'False' else 1)
    exang = exang_map[exang_label]

    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=float(default_vals['oldpeak']), step=0.1)

    slope_map = {
        "Upsloping (Normal elevation post-exercise)": "upsloping",
        "Flat (Equivocal / possible ischemia)": "flat",
        "Downsloping (High indicator of severe ischemia)": "downsloping"
    }
    slope_default_label = [k for k, v in slope_map.items() if v == default_vals['slope']][0]
    slope_label = st.selectbox("Peak Exercise ST Segment Slope", list(slope_map.keys()), index=list(slope_map.keys()).index(slope_default_label))
    slope = slope_map[slope_label]

    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy (0 - 3)", min_value=0, max_value=3, value=int(default_vals['ca']), step=1)

    thal_map = {
        "Normal (Normal blood flow)": "normal",
        "Fixed Defect (No blood flow in scar/infarct)": "fixed defect",
        "Reversible Defect (Temporary ischemia during exertion)": "reversable defect"
    }
    thal_default_label = [k for k, v in thal_map.items() if v == default_vals['thal']][0]
    thal_label = st.selectbox("Thalassemia Status", list(thal_map.keys()), index=list(thal_map.keys()).index(thal_default_label))
    thal = thal_map[thal_label]

# ----------------------------------------------------
# Prediction Execution
# ----------------------------------------------------
st.markdown("---")

predict_clicked = st.button("🔍 Analyze Cardiovascular Risk", use_container_width=True, type="primary")

if predict_clicked:
    with st.spinner("Processing clinical biomarkers and evaluating AI model..."):
        try:
            # 1. Construct raw DataFrame
            input_dict = {
                'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalch': thalch,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal
            }
            raw_df = pd.DataFrame([input_dict])

            # 2. One-hot encode categorical features
            cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
            df_encoded = pd.get_dummies(raw_df, columns=cat_cols, dtype=int)

            # 3. Align with expected features
            for col in expected_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_aligned = df_encoded[expected_cols]

            # 4. Standard scale numerical features
            scale_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
            df_aligned[scale_cols] = scaler.transform(df_aligned[scale_cols])

            # 5. PCA Transformation (13 components expected by the models)
            pca_features = [f"PCA{i+1}" for i in range(13)]
            X_pca_raw = pca.transform(df_aligned)[:, :13]
            X_pca = pd.DataFrame(X_pca_raw, columns=pca_features)

            # 6. Model Prediction
            prediction = active_model.predict(X_pca)[0]
            
            # Predict probability if supported
            has_proba = hasattr(active_model, "predict_proba")
            if has_proba:
                proba = active_model.predict_proba(X_pca)[0]
                risk_prob = proba[1]
                healthy_prob = proba[0]
            else:
                risk_prob = 1.0 if prediction == 1 else 0.0
                healthy_prob = 1.0 - risk_prob

            # ----------------------------------------------------
            # Display Results
            # ----------------------------------------------------
            st.markdown("### 📊 Diagnostic Evaluation Results")

            result_col1, result_col2 = st.columns([1, 1])

            with result_col1:
                if prediction == 1:
                    st.error("### ⚠️ Result: HIGH RISK OF HEART DISEASE")
                    st.markdown(
                        f"The model predicts a **high probability** of underlying heart disease. Clinical consultation and further cardiovascular testing (e.g., stress echocardiogram, coronary angiogram) are advised."
                    )
                else:
                    st.success("### ✅ Result: LOW RISK OF HEART DISEASE")
                    st.markdown(
                        f"The model indicates a **low probability** of heart disease based on current markers. Continue maintaining healthy lifestyle habits and routine checkups."
                    )

            with result_col2:
                st.markdown(f"**Model Used**: `{selected_model_name}`")
                if has_proba:
                    st.metric(
                        label="Heart Disease Risk Probability",
                        value=f"{risk_prob:.1%}",
                        delta=f"Healthy Probability: {healthy_prob:.1%}"
                    )
                    st.progress(float(risk_prob))
                else:
                    st.info(f"Model prediction: **{'High Risk' if prediction == 1 else 'Low Risk'}** (Binary output)")

            # Additional clinical summary table
            with st.expander("🔍 View Processed Input Data"):
                st.write("**Patient Input Summary:**")
                st.json(input_dict)
                st.write("**PCA Components (13 Features):**")
                st.dataframe(X_pca)

        except Exception as err:
            st.error(f"❌ An error occurred during prediction: {err}")
            st.exception(err)
