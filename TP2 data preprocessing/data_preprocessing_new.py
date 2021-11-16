# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#####
# Importer le dataset
#####

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#####
# Gérer les données manquantes
# donnees distribuees selon une loi normale
# absence d'outliers imputer par la moyenne
# dans le cas contraire imputer par la mediane
#####

#####
# Gérer les données manquantes
#####

# importer la classe Imputer
from sklearn.preprocessing import Imputer

# instancier la classe Imputer en precisant la strategie d'imputation

imputer = Imputer (missing_values = 'NaN', strategy = 'mean', axis = 0)
# Lire l'objet imputer aux colonnes de X à imputer 
imputer.fit(X[:, 1:3]) # lié les colonnes a l'objet d'imputation

# remplacer les valeurs manquantes par la moyenne des colonnes
X[:, 1:3] = imputer.transform(X[:, 1:3])

#####
# Gérer les variables catégoriques
#####

# les modeles ML sont basés sur des equations mathematiques
# qui implementent difficilement les variables non numerique 

# importer les classes LabelEncoder et OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# creer une instance de LabelEncoder pour la variable categorique de X
labelencoder_X = LabelEncoder()

# fiter et transformer sur la colonne a transformer
# il transforme la variable categorique a trois niveau 
# en valeurs numerique 0 - 1- 2 pour utiliser OneHotEncoder
# qui ne travail q'avec des variables numeriques

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# instancier OneHotEncoder pour l'encodager sous forme de demie variable
# 0 entre crocher pour l'indice la/des colonnes a encoder
onehotencoder = OneHotEncoder(categorical_features = [0])

# fiter puis transformer 
X = onehotencoder.fit_transform(X).toarray()

# intancier LabelEncoder pour y
labelencoder_y = LabelEncoder()
# fietr et transformer
y = labelencoder_y.fit_transform(y)

# Diviser le dataset entre le Training set et le Test set
# ? le modele de ML va apprendre le model des correlations entre la variable independante
# et la variable dependante sur un sous data set du jeu de données le tranning set (80% du jeu de donnees)
# mais pour verifier qu'il n'y a pas de sur-apprentissge ie que le modele n'a pas apris par coeur
# le modele des correlations on va le tester le test set (20% de nouvelle valeurs)

# importer la fonction train_test_split
from sklearn.model_selection import train_test_split

# test_size = 0.2
# train_size = 0.8
# random_state = 0 pour que deux executions differentes donne le meme training
# et test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# ? mettre toute les variables a la meme echelle
# pour ne pas qu'une variable ecrase l'autre dans les modeles de ML
# ex : age peut etre ecraser par le salaire dans le 
# modele de Ml
# pour le feature scaling importons la classe StandardScaler
from sklearn.preprocessing import StandardScaler
# instancier StandardScaler
sc = StandardScaler()
# fiter et transformer le training et le test set
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
