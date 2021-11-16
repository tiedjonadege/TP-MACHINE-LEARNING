# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:12:23 2019

@author: M Stan
"""
# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Representations graphiques
########
# lien
# http://www.python-simple.com/python-matplotlib/scatterplot.php
# https://matplotlib.org/3.1.1/gallery/index.html

### scarter plot ( nuage de point)
# Importer le dataset
dataset = pd.read_csv('Salary_Data.csv')
# afficher les deux premieres lignes du jeu de données
dataset.head(4)
# afficher les colonnes
dataset.columns

#######
# le scarter plot
#######

x = dataset['YearsExperience']
y = dataset['Salary']

plt.scatter(x, y)   # , edgecolors='r'
plt.xlabel('Années d\'experience')
plt.ylabel('salaire')
plt.title('Evolution du salaire')
plt.show()

#######
# Boxplot
#######

box_plot_data=[x]
plt.boxplot(box_plot_data)
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

value1 = dataset[dataset.grade=='A'].Salary
value2=dataset[dataset.grade=='B'].Salary

box_plot_data=[value1,value2]
plt.boxplot(box_plot_data,)
plt.show()

#######
# Histogrammes
#######

y = dataset['Salary']
plt.hist(y,5, histtype='bar',
align='mid', color='c', label='Salaire',edgecolor='black')
plt.legend()
plt.title('Histogramme des salairs')
plt.show()

y = dataset['Salary']
plt.hist(y,5, 
align='mid', color='c', label='Salaire',edgecolor='black', orientation = 'horizontal')
plt.legend()
plt.ylabel('Salaire')
plt.xlabel('Frequences')
plt.title('Histogramme des salairs')
plt.show()

## histogrammes supperposés des ages dans les groupes A et B
value1 = dataset[dataset.grade=='A'].Salary
value2=dataset[dataset.grade=='B'].Salary

plt.hist([value1, value1], bins = 5, color = ['yellow', 'green'],
            edgecolor = 'red', hatch = '/', label = ['A', '.B'],
            histtype = 'barstacked')
plt.ylabel('valeurs')
plt.xlabel('nombres')
plt.title('2 series superposees')
plt.legend()
plt.show()

#######
# bar plot
#######

### Barplot simple
# Creer fake dataset:
haut = [3, 12, 5, 18, 45]
bars = ('A', 'B', 'C', 'D', 'E')
y_pos = np.arange(len(bars))
# Creer les barres 
plt.bar(y_pos, haut)
# ajouter des nons a l'axe des abcisses
plt.xticks(y_pos, bars)
# afficher le graphe
plt.show()

# ou 

plt.bar(range(5), [3, 12, 5, 18, 45], width = 0.6, color = 'yellow',
  edgecolor = 'blue', linewidth = 2, 
  ecolor = 'magenta', capsize = 10)
plt.xticks(range(5), ['A', 'B', 'C', 'D', 'E'])
# afficher le graphe
plt.show()

### Barplot double
data = [[5., 25., 50., 20.],[4., 23., 51., 17.]]
X = np.arange(4)
plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
plt.show()

#######
# Pie / camembert
#######

# pie plot simple

x = [1, 2, 3, 4, 10]
plt.pie(x, labels = ['A', 'B', 'C', 'D', 'E'])
plt.show()

# pie plot simple somme des valeurs inferieur a 1
plt.figure(figsize = (8, 8))    # agrandir l'affichage
x = [0.1, 0.2, 0.3, 0.1]
plt.pie(x, labels = ['A', 'B', 'C', 'D'])
plt.legend()
plt.show()

# exemple complet
plt.figure(figsize = (8, 8))    
x = [1, 2, 3, 4, 10]
plt.pie(x, labels = ['A', 'B', 'C', 'D', 'E'],
           colors = ['red', 'green', 'yellow'],
           explode = [0, 0.2, 0, 0, 0],
           autopct = lambda x: str(round(x, 2)) + '%',
           pctdistance = 0.7, labeldistance = 1.4,
           shadow = True)
plt.legend()
plt.show()

# NB: nous pouvez utiliser ploty express pour vos representations graphiques
# https://plot.ly/python/bar-charts/

