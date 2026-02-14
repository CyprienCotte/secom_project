import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def load_and_preprocess():
    # Chargement (Slide 4)
    X = pd.read_csv('secom.data', sep='\s+', header=None)
    y = pd.read_csv('secom_labels.data', sep='\s+', header=None, names=['Result', 'Date'])
    y = y['Result'].replace(-1, 0) # Encodage 0/1 (Slide 12)

    # 1. Suppression des colonnes constantes (Slide 12)
    X = X.loc[:, X.nunique() > 1]
    
    # Split Stratifié (Support 7)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# --- Pipeline de pointe ---
# 1. Imputation (Médiane)
# 2. Standardisation (Slide 26)
# 3. Sélection des 40 meilleures features (Support Projet)
# 4. Modèle avec poids équilibrés (Support 7)
pipeline_opti = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    SelectKBest(score_func=f_classif, k=40),
    LogisticRegression(class_weight='balanced', max_iter=1000)
)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    # Entraînement
    pipeline_opti.fit(X_train, y_train)
    
    # Évaluation
    y_pred = pipeline_opti.predict(X_test)
    print("Rapport de Performance Amélioré :")
    print(classification_report(y_test, y_pred))