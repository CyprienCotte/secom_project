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

print(f"Ratio d'échecs dans train : {y_train.mean():.2%}")
print(f"Ratio d'échecs dans test : {y_test.mean():.2%}")