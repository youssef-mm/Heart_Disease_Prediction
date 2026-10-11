# ❤️ Heart Disease Prediction AI System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application designed to assess cardiovascular risk from patient clinical markers and vitals using models trained on the UCI Heart Disease dataset.

---

## 🌟 Features

- **Multi-Model Support**: Select between **XGBoost** (Recommended), **Voting Ensemble**, **Random Forest**, **SVM**, **Decision Tree**, or **Logistic Regression**.
- **Interactive Web Interface**: Built with modern Streamlit, featuring real-time risk classification, confidence probability scores, and risk gauges.
- **Quick-Test Presets**: One-click sample patient profiles (Low Risk vs High Risk) for instant demonstration and testing.
- **Robust Preprocessing Pipeline**: Clinical feature mapping, standard scaling, and 13-component Principal Component Analysis (PCA).
- **Deployment Ready**: Fully configured for **Streamlit Community Cloud**, **Render**, **Railway**, and **Docker**.

---

## 🏗️ Project Architecture

```
Heart_Disease_Prediction/
├── app.py                     # Primary Streamlit web application entrypoint
├── requirements.txt           # Production dependencies
├── Dockerfile                 # Container deployment specification
├── Procfile                   # Cloud web process entrypoint
├── .streamlit/
│   └── config.toml           # UI styling and server config
├── data/
│   ├── heart_disease.csv                  # Raw dataset (UCI)
│   ├── heart_disease_preprocessed.csv     # Cleaned and scaled data
│   └── heart_disease_preprocessed_pca.csv # PCA-transformed data
├── models/
│   ├── final_model_XGBoost_model.pkl      # Trained XGBoost model
│   ├── Voting_model.pkl                   # Voting ensemble classifier
│   ├── RF_model.pkl                       # Random Forest classifier
│   ├── SVM_model.pkl                      # Support Vector Machine
│   ├── DT_model.pkl                       # Decision Tree classifier
│   ├── log.pkl                            # Logistic Regression
│   ├── pca.pkl                            # Fitted PCA transformer
│   ├── scaler.pkl                         # Fitted StandardScaler
│   └── columns.pkl                        # Feature column alignment list
├── results/                               # Feature importance and clustering plots
├── src/                                   # Pipeline training & analysis scripts
│   ├── app.py
│   ├── data_pipeline.py
│   ├── feature_selection_pca.py
│   ├── hyperparameter_tuning.py
│   ├── train_model.py
│   └── clustering.py
└── UI/
    └── app.py
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/youssef-mm/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
```

### 2. Set up virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```
The app will open automatically at `http://localhost:8501`.

---

## ☁️ Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended & Free)
1. Fork or push this repository to your GitHub account: `https://github.com/youssef-mm/Heart_Disease_Prediction`.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository: `youssef-mm/Heart_Disease_Prediction`.
5. Set **Main file path**: `app.py`.
6. Click **"Deploy"**! Your application will be live with a public URL in seconds.

### Option 2: Render
1. Go to [render.com](https://render.com/) and create a new **Web Service**.
2. Connect your GitHub repository.
3. Select **Python** runtime:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
4. Deploy service.

### Option 3: Docker
Build and run the Docker container locally or on any cloud provider:
```bash
docker build -t heart-disease-prediction .
docker run -p 8501:8501 heart-disease-prediction
```

---

## 📊 Dataset & Features

| Feature | Description | Range / Categories |
| :--- | :--- | :--- |
| `age` | Patient age | 29 - 77 years |
| `sex` | Biological sex | Male, Female |
| `cp` | Chest pain type | Typical Angina, Atypical Angina, Non-anginal, Asymptomatic |
| `trestbps` | Resting blood pressure | 50 - 250 mmHg |
| `chol` | Serum cholesterol | 100 - 600 mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | Normal, High |
| `restecg` | Resting ECG results | Normal, ST-T Abnormality, LVH |
| `thalch` | Max heart rate achieved | 50 - 250 bpm |
| `exang` | Exercise-induced angina | Yes, No |
| `oldpeak` | ST depression induced by exercise | 0.0 - 10.0 |
| `slope` | Peak exercise ST segment slope | Upsloping, Flat, Downsloping |
| `ca` | Major vessels colored by fluoroscopy | 0 - 3 |
| `thal` | Thalassemia status | Normal, Fixed Defect, Reversible Defect |

---

## ⚕️ Medical Disclaimer
This application is designed for educational, research, and technical demonstration purposes. It is not certified for clinical diagnostic use. Always consult healthcare professionals for medical decisions.