import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss

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
    y_prob = pipeline_opti.predict_proba(X_test)
    # print("Rapport de Performance Amélioré :")
    # print(classification_report(y_test, y_pred))

# print(f"Accuracy Baseline : {accuracy_score(y_test, y_pred):.2%}")
# print(f"Log-Loss Baseline : {log_loss(y_test, y_prob):.4f}")

# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, roc_curve, auc

# # # 1. Matrice de Confusion
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Matrice de Confusion : Baseline')
# plt.ylabel('Réalité')
# plt.xlabel('Prédiction')
# plt.show()

# # # 2. Courbe ROC (Slide 14)
# fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.title('Courbe ROC - Performance du Modèle')
# plt.legend()
# plt.show()