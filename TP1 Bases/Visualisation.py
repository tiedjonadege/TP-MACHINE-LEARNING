# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:12:23 2019

@author: M Stan
"""
import pandas as pd
import numpy as np
import scipy as np
#from plotly.offline import iplot, init_notebook_node
#init.notebook.node()

#import plotly.graph_objs as go
# importer son jeu de doonnées
df = pd.read_excel("Comorbid.xlsx")
# afficher les deux premieres lignes du jeu de données
df.head(2)
# afficher les colonnes
df.columns

## parametres de position
# la moyenne
df.Age.mean()
#la mediane
df.Age.median()
#le mode 
df.DiabetesMellitus.mode()
df.DiabetesMellitus.value_counts()
df.DiabetesMellitus.value_counts(normalize = True)*100
# les quantiles
df.Age.quantile([0, 0.25,0.5, 0.75, 0.9, 1])
np.percentile(df.Age.values,[0, 25,50, 75, 90, 100])
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
