import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, balanced_accuracy_score
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Correction du SyntaxWarning (Utilisation de r'')
X = pd.read_csv('secom.data', sep=r'\s+', header=None)
y = pd.read_csv('secom_labels.data', sep=r'\s+', header=None, names=['Result', 'Date'])
y = y['Result'].replace(-1, 0)

# Split Stratifié (Pillar 1 du support Imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline optimisé avec SMOTE [cite: 8, 12, 115]
pipeline_opti = make_pipeline_imb(
    SimpleImputer(strategy='median'), # Gestion des NaNs [cite: 12]
    StandardScaler(),                 # Feature Scaling [cite: 392]
    SMOTE(random_state=42),           # Sur-échantillonnage synthétique
    SelectKBest(score_func=f_classif, k=40), # Sélection des 40 meilleures features [cite: 453]
    LogisticRegression(max_iter=1000) # Algorithme de classification linéaire [cite: 49]
)

if __name__ == "__main__":
    pipeline_opti.fit(X_train, y_train)
    y_pred = pipeline_opti.predict(X_test)
    y_prob = pipeline_opti.predict_proba(X_test)
#     print(f"Nouveau Balanced Accuracy (SMOTE) : {balanced_accuracy_score(y_test, y_pred):.4f}")
#     print("Rapport de Performance Amélioré :")
#     print(classification_report(y_test, y_pred))

# print(f"Accuracy Baseline : {accuracy_score(y_test, y_pred):.2%}")
# print(f"Log-Loss Baseline : {log_loss(y_test, y_prob):.4f}")



# # # # 1. Matrice de Confusion
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Matrice de Confusion : Baseline')
# plt.ylabel('Réalité')
# plt.xlabel('Prédiction')
# plt.show()

# # # # 2. Courbe ROC (Slide 14)
# fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.title('Courbe ROC - Performance du Modèle')
# plt.legend()
# plt.show()





# 1. Récupérer les noms des colonnes après le nettoyage initial
feature_names = X_train.columns

# 2. Récupérer le masque des colonnes sélectionnées par SelectKBest
mask = pipeline_opti.named_steps['selectkbest'].get_support()
selected_features = feature_names[mask]

# 3. Récupérer les coefficients du modèle
coefs = pipeline_opti.named_steps['logisticregression'].coef_[0]

# 4. Créer un DataFrame pour la visualisation
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': coefs,
    'Odds_Ratio': np.exp(coefs) # Calcul de l'Odds Ratio (Slide 18)
})

# 5. Trier par impact (coefficients les plus élevés = plus de risque)
top_5_critical = importance_df.sort_values(by='Coefficient', ascending=False).head(5)

print("\n--- TOP 5 DES CAPTEURS LES PLUS CRITIQUES ---")
print(top_5_critical[['Feature', 'Coefficient', 'Odds_Ratio']])

import shap

# 1. On prépare les données (il faut qu'elles soient passées par le scaler/imputer)
# On récupère le transformateur du pipeline (sans le modèle final)
preprocessor = pipeline_opti[:-1]
X_test_transformed = preprocessor.transform(X_test)
feature_names = X_train.columns[pipeline_opti.named_steps['selectkbest'].get_support()]

# 2. Création de l'Explainer pour la Régression Logistique
explainer = shap.LinearExplainer(
    pipeline_opti.named_steps['logisticregression'], 
    X_test_transformed
)
shap_values = explainer.shap_values(X_test_transformed)

# 3. Visualisation (Summary Plot)
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)