#!/usr/bin/env python
# coding: utf-8

# In[73]:


# Importer la base de données 

import pandas as pd
df = pd.read_csv("/Users/fadilamoussaoui/Desktop/Session H22/PSY4016 - Programmation/Projet/Base_de_données_travail.csv", sep = ";")

df


# In[74]:


# Nom des colonnes de la base de données 

df.columns


# In[75]:


# Enelever les colonnes inutiles pour le projet 

df_projet = df.drop(columns =['Age','Salary','Study_Level','Mood_Disorder','BMI','Sport','Music',
                              'Extraversion','Conscientiousness','Instability.Neuroticism','Openness_to_Experience.Intellect',
                            'Honesty.Humility','Detachment','Psychoticism','Negative_Affect','Antagonism'])

df_projet


# In[76]:


# Nom des colonnes de la base de données utiles au projet

df_projet.columns


# In[77]:


# Lire les données de la colonne "Sex"

for value in df_projet["Sex"]:
      print(value)


# In[78]:


# Lire les données de la colonne "Meditation"

for value in df_projet["Meditation"]:
      print(value)


# In[79]:


# Lire les données de la colonne "Empathy.Agreeableness"
# !!! Il y a une donnée manquante 

for value in df_projet["Empathy.Agreeableness"]:
      print(value)
        


# In[80]:


# Lire les données de la colonne "Disinhibition"

for value in df_projet["Disinhibition"]:
      print(value)
    


# In[87]:


# Coder la variable Méditation ( "No" = 0; "Yes" = 1)
for value in df_projet['Meditation'] : 
    if value == "No":
        df_projet.replace(value, int(str(0)), inplace = True)
        
    if value == "Yes":
        df_projet.replace(value, int(str(1)), inplace = True)
            
# Lire les données de la colonne "Meditation"

for value in df_projet["Meditation"]:
      print(value)


# In[90]:


# Type de variables 

print(type("Sex"))
print(type("Meditation"))
print(type("Empathy.Agreeableness"))
print(type("Disinhibition"))


# In[ ]:


# Transformer de chaîne à float

for col in df.columns:
    if Meditation in col:
        df_projet[Meditation] = df_projet[Meditation].astype(float)
    if Empathy.Agreeableness in col:
        df_projet[Empathy.Agreeableness] = df_projet[Empathy.Agreeableness].astype(float)
    if Disinhibition in col:
        df_projet[Disinhibition] = df_projet[Disinhibition].astype(float)


# In[46]:


# Gérer les données manquantes pour la colonne "Empathy.Agreeableness"

import sklearn 
import seaborn
from sklearn import impute
import numpy as np

imp = sklearn.impute.SimpleImputer(missing_values = np.nan, strategy = "mean")


# In[ ]:


# Modification des valeurs manquantes pour la moyenne des scores
import numpy as np
import sklearn 
from sklearn import impute
imp = sklearn.impute.SimpleImputer(missing_values = np.nan, strategy = "mean")

for col in [Empathy.Agreeableness]:
   
    for index in range(len(df)):
        
        for i in [df_projet[col][index]]: 
                cell =  i  
                imp.fit(df[[col]])
                df[col] = imp.transform(df[[col]])


# In[101]:


# Créer les scores Z

def create_scoreZ():

    import numpy as np
    import sklearn 
    from sklearn import preprocessing
    import pandas as pd
    scaler = sklearn.preprocessing.StandardScaler()



    columns = df_projet.columns.values.tolist()

    for col in df_projet.columns : 
            
            for i in df[col]:  
                x_value = df_projet[col].to_numpy() 
                x_value = x_value[:, np.newaxis] 
                scaled_value = scaler.fit(x_value) 
                scaled_value = scaler.transform(x_value)
            
            df['Z_'+col] = scaled_value


# In[102]:


# Test T pour hypothèse 1
    
import scipy
from scipy import stats
    
avec_meditation = df_projet[df_projet['Meditation'] == 'Yes']["Empathy.Agreeableness"]
sans_meditation = df_projet[df_projet['Meditation'] == 'No']["Empathy.Agreeableness"]

stats.ttest_ind(avec_meditation, sans_meditation)

res = stats.ttest_ind(avec_meditation, sans_meditation)
print(res.statistic, res.pvalue)


# In[ ]:


# Graphique pour hypothèse 1

import seaborn as sns
with sns.axes_style(style='ticks'):
    g = sns.catplot(data=df_projet, x = 'Meditation', y = 'Empathy.Agreeableness', kind='box')
    g.set_axis_labels('Meditation', 'Empathy.Agreeableness')
    


# In[44]:


# Test T pour hypothèse 2

avec_meditation = df_projet[df_projet['Meditation'] == 'Yes']["Disinhibition"]
sans_meditation = df_projet[df_projet['Meditation'] == 'No']["Disinhibition"]

stats.ttest_ind(avec_meditation, sans_meditation)

res = stats.ttest_ind(avec_meditation, sans_meditation)
print(res.statistic, res.pvalue)


# In[ ]:


# Graphique pour hypothèse 2 

with sns.axes_style(style='ticks'):
    sns.catplot(data=df_projet, x = 'Meditation', y = 'Disinhibition', kind='box')
    set_axis_labels('Meditation', 'Disinhibition') 


# In[104]:


# Préalables AA supervisé

import pandas as pd
import numpy as np
import scipy
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,5) 
import seaborn as sns;
import sklearn
from sklearn.datasets import fetch_lfw_people, make_blobs, make_circles, load_digits
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC        
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from IPython.display import Image
from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed
from math import sqrt


# In[ ]:


# AA supervisé code 

class KNN_GRAPHIQUE:

	def AA_supervise():

		X_data = pd.read_excel("/Users/fadilamoussaoui/Desktop/Session H22/PSY4016 - Programmation/Projet/Base_de_données_travail.csv")
		X = X_data['Sex', 'Meditation', 'Empathy.Agreeableness', 'Disinhibition'].values

		Meditation = X_data.Meditation.to_list()
		le = LabelEncoder()
		label=le.fit_transform(Meditation)
		y = label

		knn = KNN(n_neighbors=3)
		y_pred = cross_validation_predict(knn, X, y, cv = 5)

		print(sqrt(mean_squared_error(y,y_pred)))
		print(r2_score(y,y_pred))

		error = []
		for k in range(1,100):
		    knn = KNN(n_neighbors=k)
		    y_pred = cross_validation_predict(knn, X, y)
		    error.append(mean_squared_error(y,y_pred))
            


# In[ ]:


# AA supervisé graphique

ax = plt.axes()
ax.plot(range(1,100),error, color="red", linestyle="-", marker="o",
         markerfacecolor="blue", markersize=10)
ax.set_title("KNN non standardisé")
ax.set_xlabel("K")
ax.set_ylabel("Erreur")
plt.show()


# In[ ]:


# Préalables AA non-supervisé
import sklearn
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
print('sklearn version:', sklearn.__version__)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA


# In[ ]:


# AA non-supervisé code 

def AA_non_supervise():

    X_data = pd.read_excel("/Users/fadilamoussaoui/Desktop/Session H22/PSY4016 - Programmation/Projet/Base_de_données_travail.csv")
    X = X_data['Sex', 'Meditation', 'Empathy.Agreeableness', 'Disinhibition'].values

    x = X_data.loc[:, caractéristiques].values
    y = X_data.loc[:,["Meditation"]].values

    x = StandardScaler().fit_transform(x)


    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    Df1 = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])
    Df2 = pd.concat([Df1, X_data[['Meditation']]], axis = 1)


# In[ ]:


# AA non-supervisé graphique 

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['No', 'Yes']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Meditation'] == target
    ax.scatter(Df2.loc[indicesToKeep, 'principal component 1']
               , Df2.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    
print(pca.explained_variance_ratio_)


# In[ ]:


# Base de données sqlite avec une pipeline

import sqlite
sqlite.try_sqlite()

