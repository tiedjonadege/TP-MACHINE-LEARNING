# Regression Linear Simple

# Importer les libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# extraire la matrice des variables independantes
# et le vecteur des variables depedentes
 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitter the dataset : Training set  2/3 et Test set (1/3)
# random_state = 0 pour obtenir les memes jeux de données

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Pour la regression lineaire simple il n'y a pas de normalisation (Feature Scaling)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Lier cet objet à notre training set (X_train, y_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)#il a appris les corrélation direct ici

# Predire les resultatsdu Test set 
y_pred = regressor.predict(X_test)
# regressor.predict(15)

# Visualiser les results du Training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salaire / Experience (Training set)')
plt.xlabel('Année d\'experience')
plt.ylabel('Salaire')
plt.show()

# Visualiser les results du Test set 
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salaire / Experience (Training set)')
plt.xlabel('Année d\'experience')
plt.ylabel('Salaire')
plt.show()

# Calculer la precision du modele
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


