import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, roc_curve, auc, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

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

X_train, X_test, y_train, y_test = load_and_preprocess()


# Définition des modèles à tester
# models = {
#     "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
#     "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100),
#     "XGBoost": XGBClassifier(scale_pos_weight=14) # Ratio 1463/104 approx 14
# }

# # Boucle d'évaluation
# for name, model in models.items():
#     # On réutilise ton pipeline avec SelectKBest (40)
#     pipeline = make_pipeline(
#         SimpleImputer(strategy='median'),
#         StandardScaler(),
#         SelectKBest(score_func=f_classif, k=40),
#         model
#     )

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, balanced_accuracy_score


# Définition des modèles à tester (Slide 7 & Lecture 6)
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=14, random_state=42) # scale_pos_weight ~ 1463/104
}

 # Scoreur personnalisé : Balanced Accuracy (Slide 21 - Support 7)
 # Note : BER = 1 - Balanced Accuracy
scorer = make_scorer(balanced_accuracy_score)

print("Résultats de la Validation Croisée (10-folds) :")
for name, model in models.items():
#     # Pipeline complet pour chaque modèle
    pipeline = make_pipeline(
         SimpleImputer(strategy='median'),
         StandardScaler(),
         SelectKBest(score_func=f_classif, k=40),
         model
     )
    
     # Cross-validation à 10 plis comme recommandé par le projet
    scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring=scorer)
    print(f"{name} : {scores.mean():.2%} (+/- {scores.std():.2%})")   
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    
    print(f"--- {name} ---")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy {name} : {accuracy_score(y_test, y_pred):.2%}")
    print(f"Log-Loss {name} : {log_loss(y_test, y_prob):.4f}")


    # # 1. Matrice de Confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion : {name}')
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    plt.show()

    # # 2. Courbe ROC (Slide 14)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Courbe ROC - Performance du Modèle')
    plt.legend()
    plt.show()

