# print("hello, this is my first test with micro")

import pandas as pd 

def load_data():

    X = pd.read_csv('secom.data', sep = '\s+', header = None)

    y = pd.read_csv('secom_labels.data', sep = '\s+', header = None, names = ['Result', 'Timestamp'])

    # print("Structure de X :", X.shape)
    # print("Structure de y :", y.shape)

    # # Vérification du déséquilibre des classes
    # print("\nDistribution des classes :")
    # print(y['Result'].value_counts())

    return X, y 

if __name__ == "__main__" :     
    X, y = load_data()

y_encoded = y['Result'].replace(-1, 0)

# Split du dataset en train et test en utilisant le Stratified Sampling

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# print(f"Ratio d'échecs dans train : {y_train.mean():.2%}")
# print(f"Ratio d'échecs dans test : {y_test.mean():.2%}")

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# 1. Imputation : on remplace les NaNs par la médiane (robuste aux outliers)
# 2. Standardisation : on applique (X - mu) / sigma 
preprocessor = make_pipeline(
    SimpleImputer(strategy='median'), 
    StandardScaler()
)

# Application sur le jeu d'entraînement
# On "fit" (apprend la médiane et l'écart-type) ET on transforme
X_train_prepared = preprocessor.fit_transform(X_train)

# Application sur le jeu de test
# On utilise uniquement .transform() pour ne pas "fuiter" d'infos du test
X_test_prepared = preprocessor.transform(X_test)

# print(f"Données d'entraînement préparées : {X_train_prepared.shape}")
# print(f"Données de test préparées : {X_test_prepared.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# 1. Initialisation du modèle 
# On utilise la configuration par défaut pour cette Baseline
model = LogisticRegression(max_iter=1000)

# 2. Entraînement 
# On utilise les données préparées (Imputed + Scaled)
model.fit(X_train_prepared, y_train)

# 3. Prédictions 
y_pred = model.predict(X_test_prepared)
y_prob = model.predict_proba(X_test_prepared)

# 4. Évaluation rapide 
print(f"Accuracy Baseline : {accuracy_score(y_test, y_pred):.2%}")
print(f"Log-Loss Baseline : {log_loss(y_test, y_prob):.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# 1. Matrice de Confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion : Baseline')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.show()

# 2. Courbe ROC (Slide 14)
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Courbe ROC - Performance du Modèle')
plt.legend()
plt.show()