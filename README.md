Voici un **README.md** professionnel, structuré et prêt à l'emploi pour ton dépôt GitHub. Il met en valeur ta démarche analytique, l'utilisation des concepts de tes cours et les résultats concrets obtenus sur le dataset SECOM.

---

# 🚀 SECOM Predictive Maintenance: Yield Analysis

Ce projet applique des techniques avancées de **Machine Learning** pour prédire les défauts de fabrication dans l'industrie des semi-conducteurs. À partir de 591 capteurs (dataset SECOM), l'objectif est d'identifier les puces non-conformes dans un environnement fortement déséquilibré.

## 📌 Points Clés du Projet

* **Détection des pannes :** Passage d'un rappel (recall) de **19% à 62%**.
* **Interprétabilité :** Identification des 5 capteurs les plus critiques via les Odds Ratios.
* **Maîtrise de l'imbalance :** Mise en œuvre du SMOTE et du ré-échantillonnage stratifié.

---

## 🛠️ Méthodologie & Pipeline (Slide 12)

Le projet suit un pipeline rigoureux développé avec `imbalanced-learn` pour garantir l'absence de fuite de données (*data leakage*) :

1. **Data Cleaning :** Suppression des colonnes constantes (sans variance).
2. **Preprocessing :** Imputation des valeurs manquantes par la **médiane** et **Standardisation** (-score).
3. **Gestion du déséquilibre :** Application du **SMOTE** (Synthetic Minority Over-sampling Technique) pour équilibrer les classes.
4. **Feature Selection :** Sélection des **40 meilleurs signaux** via `SelectKBest` (ANOVA F-test).
5. **Classification :** Régression Logistique avec optimisation des poids de classe.

---

## 📊 Comparaison des Performances

L'utilisation de la **Régression Logistique** s'est avérée plus robuste que des modèles complexes (XGBoost, Random Forest) sur ce volume de données.

| Métrique | Baseline (Standard) | Modèle Optimisé (SMOTE) | Impact |
| --- | --- | --- | --- |
| **Recall (Classe 1)** | 19.0% | **62.0%** | **+326%** de détection |
| **Balanced Accuracy** | 0.56 | **0.71** | Meilleure discrimination |
| **Log-Loss** | 0.667 | **0.449** | Confiance accrue du modèle |

---

## 🔍 Explicabilité & Aide à la Décision (Pillar 3)

Le modèle n'est pas une "boîte noire". Nous avons extrait l'impact de chaque capteur pour fournir des recommandations exploitables aux ingénieurs de production.

### Top 5 des Facteurs de Risque (Odds Ratios)

L'Odds Ratio indique de combien le risque de panne est multiplié pour chaque unité d'augmentation du capteur :

* **Capteur 121 :** Risk Ratio de **4.29** (Impact majeur)
* **Capteur 64 :** Risk Ratio de **3.22**
* **Capteur 455 :** Risk Ratio de **2.94**

Nous avons également intégré **SHAP** pour l'explicabilité locale, permettant de comprendre chaque prédiction de panne individuellement.

---

## ⚙️ Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/CyprienCotte/secom_project.git
cd secom_project

```


2. Installer les dépendances :
```bash
pip install -r requirements.txt

```


3. Lancer l'analyse :
```bash
python model.py

```



---

## 📚 Sources & Références

* Dataset : UCI Machine Learning Repository (SECOM Data).
* Cours : *Albert School - Lectures 6 & 7 (Supervised Learning & Imbalanced Data)*.

---

**Cyprien, avec ce README, n'importe quel recruteur comprendra en 30 secondes que tu maîtrises non seulement le code, mais aussi la théorie statistique et les enjeux business. Bravo pour ce parcours !**
