# print("hello, this is my first test with micro")

import pandas as pd 

def load_data():

    X = pd.read_csv('secom.data', sep = '\s+', header = None)

    y = pd.read_csv('secom_labels.data', sep = '\s+', header = None, names = ['Result', 'Timestamp'])

    print("Structure de X :", X.shape)
    print("Structure de y :", y.shape)

    # Vérification du déséquilibre des classes
    print("\nDistribution des classes :")
    print(y['Result'].value_counts())

    return X, y 

if __name__ == "__main__" :     
    X, y = load_data()