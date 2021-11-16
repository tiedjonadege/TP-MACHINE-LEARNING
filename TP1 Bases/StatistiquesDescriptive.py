# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:12:23 2019

@author: M Stan
"""
# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

########
## statistiques descriptives
########
# importer son jeu de doonnées
df = pd.read_excel("Comorbid.xlsx")
# afficher les deux premieres lignes du jeu de données
df.head(4)
# afficher les colonnes
df.columns

## parametres de position
# la moyenne
df.Age
df.Age.mean()
#la mediane
df.Age.median()
#le mode 
df.DiabetesMellitus.mode()
df.DiabetesMellitus.value_counts()
df.DiabetesMellitus.value_counts(normalize = True)*100
# les quantiles
df.Age.quantile([0, 0.25,0.5, 0.75, 0.9, 1])
np.percentile(df.Age.values,[0, 20, 25,50, 75, 90, 100])
# la moyenne des ages des patients de grade A
df[df.Grade=='A']
df[df.Grade=='A'].Age
df[df.Grade=='A'].Age.mean()
# le,maximum et le minimum
df[df.Grade=='A'].Age.min()
df[df.Grade=='A'].Age.max()
## parametres de dispersion
df.Age.std()
df.Age.var()
df.Age.var()

# l'intervalle interquartle
from scipy import stats
stats.iqr(df.Age)

# statistique descriptive avec describe
df.Age.describe()

########
## Representations graphiques
########
# lien
# http://www.python-simple.com/python-matplotlib/scatterplot.php

### scarter plot ( nuage de point)
# Importer le dataset
dataset = pd.read_csv('Salary_Data.csv')
# afficher les deux premieres lignes du jeu de données
dataset.head(4)
# afficher les colonnes
dataset.columns

# le scarter plot
x = dataset['YearsExperience']
y = dataset['Salary']

plt.scatter(x, y)   # , edgecolors='r'
plt.xlabel('Années d\'experience')
plt.ylabel('salaire')
plt.title('Evolution du salaire')
plt.show()

# Boxplot
box_plot_data=[x]
plt.boxplot(box_plot_data)
plt.show()

box_plot_data=[x]
plt.boxplot(box_plot_data,labels=['Années d\'experience'])
plt.show()

box_plot_data=[x]
plt.boxplot(box_plot_data,labels=['Années d\'experience'])
plt.show()

box_plot_data=[x]
plt.boxplot(box_plot_data,patch_artist=True,labels=['Années d\'experience'])
plt.show()

box_plot_data=[x]
plt.boxplot(box_plot_data,patch_artist=True,labels=['Années d\'experience'],vert=0)
plt.show()

value1 = df[df.Grade=='A'].Age
value2=df[df.Grade=='B'].Age

box_plot_data=[value1,value2]
plt.boxplot(box_plot_data)
plt.show()

# Histogrammes

y = dataset['Salary']
plt.hist(y,5, histtype='bar',
align='mid', color='c', label='Salaire',edgecolor='black')
plt.legend()
plt.title('Histogramme des salairs')
plt.show()

y = dataset['Salary']
plt.hist(y,5, histtype='bar',
align='mid', color='c', label='Salaire',edgecolor='black', orientation = 'horizontal')
plt.legend()
plt.ylabel('Salaire')
plt.xlabel('frequences')
plt.title('Histogramme des salairs')
plt.show()

## histogrammes supperposés des ages dans les groupes A et B
value1 = df[df.Grade=='A'].Age
value2=df[df.Grade=='B'].Age

plt.hist([value1, value1], bins = 5, color = ['yellow', 'green'],
            edgecolor = 'red', hatch = '/', label = ['A', '.B'],
            histtype = 'barstacked')
plt.ylabel('valeurs')
plt.xlabel('nombres')
plt.title('2 series superposees')
plt.legend()
plt.show()

# bar plot






