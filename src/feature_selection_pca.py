# src/feature_selection_pca.py
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

DATA_TRAIN = "data/heart_disease_preprocessed.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run():
    df = pd.read_csv(DATA_TRAIN)
    X = df.drop(columns=['num'])
    y = df['num']

    # Feature Importance
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.plot(kind='bar', figsize=(10,5))
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/feature_importance.png")
    plt.close()
    
    # RFE
    selector = RFE(rf, n_features_to_select=5)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    with open(f"{RESULTS_DIR}/selected_features.txt", "w") as f:
        f.write("\n".join(selected_features))
    
    # PCA
    pca = PCA(n_components=min(len(X.columns), len(X)))
    pca.fit(X)
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
    plt.title("PCA Explained Variance")
    plt.xlabel("Components")
    plt.ylabel("Variance Ratio")
    plt.savefig(f"{RESULTS_DIR}/pca_variance.png")
    plt.close()
    
    # Save PCA
    import joblib
    joblib.dump(pca, 'models/pca.pkl')

    print("✅ Feature selection and PCA completed.")

if __name__ == "__main__":
    run()