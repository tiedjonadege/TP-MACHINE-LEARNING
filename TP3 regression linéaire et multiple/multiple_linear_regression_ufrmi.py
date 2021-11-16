#  Regression Lineaire Multiple 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

# description du jeu de donnes 
dataset.columns
 taille = dataset.shape

print()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoder la variable categorielle state (N° 4)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Eviter le piege des Dummy Variable 
X = X[:, 1:]

# Splitter le  dataset en  Training set(80%) and Test set (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# pas besoin de faire du Feature Scaling car l'objet regressor
# le fera automatiquement pour nous

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Appliquer notre modele de Regression Lineaire Multiple au Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predire les resultats du  Test set 
y_pred = regressor.predict(X_test)

# constrire un modele optimal en selectionnant uniquement les variable independantes
# qui ont un impact significatif sur le modele

# Backward Elimination avec p-values :
import statsmodels.formula.api as sm
X=np.append(arr = np.ones((50,1)).astype(int),values = X,axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.ols(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.ols(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

"""
"""

