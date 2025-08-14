# Heart Disease - ML Full Pipeline

## Overview
This project implements a complete ML pipeline on the Heart Disease (UCI) dataset:
- Preprocessing, EDA
- Feature selection (RFE, RandomForest importance, Chi-square)
- PCA
- Supervised models: Logistic Regression, Decision Tree, Random Forest, SVM
- Unsupervised: KMeans and Hierarchical Clustering
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Streamlit app for prediction
- Save/load model pipeline (.pkl)

## Steps
1. Put `heart_disease.csv` into `data/`
2. Create virtualenv and install packages:
   `pip install -r requirements.txt`
3. Run preprocessing:
   `python src/data_pipeline.py`
4. Run training:
   `python src/train_models.py`
5. Start Streamlit:
   `streamlit run src/app.py`

## Dataset
Download the Heart Disease dataset from UCI and save as `data/heart_disease.csv`.
Columns expected:
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

## Notes
- Models are saved to `models/`.
- Preprocessing pipeline saved to `models/preprocessor.pkl`.