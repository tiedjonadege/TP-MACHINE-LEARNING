# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:16:30 2019


"""

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Construction du modÃ¨le
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Faire de nouvelles prÃ©dictions
y_pred = classifier.predict(X_test)

# Matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Validation par la courbe CAP
################################
           
# length of the test data 
total = len(y_test) 

# Counting '1' labels in test data 
one_count = np.sum(y_test) 

# counting '0' lables in test data 
zero_count = total - one_count 

plt.figure(figsize = (10, 6)) 

# x-axis ranges from 0 to total people contacted 
# y-axis ranges from 0 to the total positive outcomes. 

plt.plot([0, total], [0, one_count], c = 'b', 
		linestyle = '--', label = 'Random Model') 
plt.legend() 

#Random Forest Classifier Line
lm = [y for _, y in sorted(zip(y_pred, y_test), reverse = True)] 
x = np.arange(0, total + 1) 
y = np.append([0], np.cumsum(lm)) 
plt.plot(x, y, c = 'b', label = 'Random classifier', linewidth = 2) 

#Perfect Model
plt.plot([0, one_count, total], [0, one_count, one_count], 
		c = 'grey', linewidth = 2, label = 'Perfect Model') 

# Visualiser les rÃ©sultats
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('RÃ©sultats du Training set')
