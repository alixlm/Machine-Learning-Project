#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project : Electricity Price Explanation

# # I. Data preprocessing
# ## 1. Import

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import spearmanr

from sklearn.model_selection import KFold


# ## 2. Data importation and basic visualization

# In[2]:


X_df =pd.read_csv('X_train_NHkHMNU.csv', delimiter= ',')
y_df =pd.read_csv('y_train_ZAN5mwg.csv', delimiter= ',')
X_test_df =pd.read_csv('X_test_final.csv', delimiter= ',')
df = pd.merge(X_df,y_df,on='ID')
print(df.shape)
df.head(10)


# In[3]:


df.isnull().sum()


# In[4]:


df.sort_values("DAY_ID")


# In[5]:


# data types
df.dtypes


# Je regarde le poucentage de valeur manquante de chaque colonne

# In[6]:


nb_missing = df.isna().sum()
rate_missing = nb_missing / df.ID.nunique()
fig, ax = plt.subplots(figsize=(4,6))
ax1 = ax
rate_missing.plot(kind="barh", ax=ax1)
ax1.grid()


# Je vérifie qu'il n'y ai pas de doublons

# In[7]:


doublons = df.duplicated()
print(doublons.sum())


# Je check la distribution de chaque feature 

# In[8]:


features = [feature for feature in df.columns if feature != "COUNTRY"]

nb_col = 6
nb_row = - (-len(features)//6)
fig, ax = plt.subplots(nb_row, nb_col, figsize=(14,14))

for i, feature in enumerate(features):
    i_col = i % nb_col
    i_row = i // nb_col
    ax1 = ax[i_row, i_col]
    
    ax1.set_title(feature)
    ax1.grid()
    df[feature].hist(bins= 30, ax=ax1, alpha=0.7)

plt.tight_layout()

fig, ax = plt.subplots()
ax1 = ax
y_df["TARGET"].hist(bins= 30, ax=ax1, alpha=0.7)
ax1.set_title("TARGET")


# On separe la data en 2 partie pour voir la distribution des features de la France et de l'Allemagne de maniere independante.

# In[9]:


features = [feature for feature in df.columns if feature != "COUNTRY"]

nb_col = 6
nb_row = - (-len(features)//6)
fig, ax = plt.subplots(nb_row, nb_col, figsize=(14,14))

print("Nb of data points by country:")
print(df.COUNTRY.value_counts())

for i, feature in enumerate(features):
    i_col = i % nb_col
    i_row = i // nb_col
    ax1 = ax[i_row, i_col]
    
    ax1.set_title(feature)
    ax1.grid()
    df[df.COUNTRY == "FR"][feature].hist(bins= 30, ax=ax1, alpha=0.7, label= "FR")
    df[df.COUNTRY == "DE"][feature].hist(bins= 30, ax=ax1, alpha=0.7, label= "DE")
    ax1.legend()

plt.tight_layout()


# In[10]:


# Check time series
features = [feature for feature in df.columns if not feature in ["COUNTRY", "DAY_ID"]]

nb_col = 6
nb_row = - (-len(features)//6)
fig, ax = plt.subplots(nb_row, nb_col, figsize=(14,14))

print("Nb of data points by country:")
print(df.COUNTRY.value_counts())

for i, feature in enumerate(features):
    i_col = i % nb_col
    i_row = i // nb_col
    ax1 = ax[i_row, i_col]
    
    ax1.set_title(feature)
    time_series = lambda COUNTRY: df[df.COUNTRY == COUNTRY].set_index("DAY_ID")[feature].sort_index()
    ax1.plot(time_series("FR"), label = "FR", alpha=0.7)
    ax1.plot(time_series("DE"), label = "DE", alpha=0.7)
    ax1.legend()
    ax1.grid()

plt.tight_layout()


# In[11]:


# missing days?
F = df.DAY_ID.value_counts().sort_index()
F = F.reindex(range(F.index.max()))
F = F.fillna(0)
F = F.value_counts()
F /= F.sum()
fig, ax = plt.subplots(figsize=(4,1))
ax1 = ax
ax1.set_title("Number of points per day in time period")
F.plot(kind="barh", ax=ax1)
ax1.grid()


# In[12]:



# Création du boxplot pour la variable cible "TARGET" par pays
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='COUNTRY', y='TARGET')
plt.title("Distribution de la variable cible 'TARGET' par pays")
plt.xlabel("Pays")
plt.ylabel("Valeur de TARGET")
plt.grid(True)
plt.show()


# On peut voir que la médiane pour les deux pays sont assez similaire et proche de 0 ce qui suggere que les valeur target sont centrées.
# On remarque également qu'il semble y avoir de nombreux outlier (valeurs extrêmes) pour la France.

# In[13]:


# Comptage des échantillons par pays
country_counts = df['COUNTRY'].value_counts()

# Création du camembert
plt.figure(figsize=(6, 6))
plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Répartition des échantillons par pays")
plt.show()


# In[14]:


# Plotting the time series of the TARGET variable
plt.figure(figsize=(14, 5))
sns.lineplot(x='DAY_ID', y='TARGET', data=df)
plt.title('Time Series Analysis of TARGET')
plt.xlabel('Day ID')
plt.ylabel('TARGET (Daily Price Variation)')
plt.show()


# ## 3. Data preprocessing

# In[15]:


def preprocess_df(df):
    columns_to_fill = [
    "DE_FR_EXCHANGE", "FR_DE_EXCHANGE", "DE_NET_EXPORT", "FR_NET_EXPORT", 
    "DE_NET_IMPORT", "FR_NET_IMPORT", "DE_RAIN", "FR_RAIN", 
    "DE_WIND", "FR_WIND", "DE_TEMP", "FR_TEMP"
    ]
    for column in columns_to_fill:
        df[column].fillna(df[column].mean(), inplace=True)
    
    df['COUNTRY'] = df['COUNTRY'].apply(lambda x: 1 if x == 'FR' else 0)
   
    return df

print(df.dtypes)


# ## 4. Data visualization

# In[16]:


def correlation_colonne (df_train_processed,seuil):
    correlations= df_train_processed.corrwith(df_train_processed['TARGET']).abs()
    #On trie les corrélations par ordre décroissant
    correlations= correlations.sort_values(ascending=False)
    correlated_columns = correlations[correlations >seuil].index
    #On les mets dans df_train_bis une dataframe que l'on utilisera pour entrainer le modèle
    df_train_bis= df_train_processed[correlated_columns]
    #On affiche les 10 colonnes avec le plus de corrélation avec le log_price
    print(correlations.sort_values(ascending=False).head(10))
    #On s'assure que toutes les colonnes utilisées pendant l'entraînement sont présentes
    return df_train_bis, correlated_columns

seuil=0.0001
df_train_processed = preprocess_df(df)


# In[17]:


correlations= df_train_processed.select_dtypes(include=['float64', 'int64', 'bool']).corrwith(df_train_processed['TARGET']).abs()
#on augmente le seuil pour prendre moins de colonne et avoir quelque chose de lisible
correlated_columns = correlations[correlations >seuil].index
#On les mets dans df_train_3 spécialemént pour faire une matrice de corrélation avec les colonnes les plus corrélée avec TARGET
df_train_3= df_train_processed[correlated_columns]

correlation_matrix = df_train_3.corr(method='pearson', min_periods=1)
non_numeric_cols = correlation_matrix.select_dtypes(exclude=[np.number]).columns
plt.figure(figsize=(18, 14))
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.show()


# # II Model choice
# ## 5. Pipeline construction to test various model

# In[18]:


def Pipeline(scaler, model):
    my_pipeline = make_pipeline(scaler, model)
    return my_pipeline


# In[19]:


def Sep_Train_Evaluation(X,y,pipeline):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  pipeline.fit(X_train,y_train)
  y_pred=pipeline.predict(X_test)
  # Calculer le score Spearman
  print(f"Spearman score: {spearmanr(y_test,y_pred).correlation}")


# In[20]:


#we are gonna use this pipelines
std_scaler = StandardScaler()
l_model = LinearRegression()
pipe_l = Pipeline(std_scaler,l_model)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
pipe_rf = Pipeline(std_scaler,rf_model)

gb_model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.2,max_depth=3,random_state=42)
pipe_gb = Pipeline(std_scaler,gb_model)


# In[21]:


df.head(10)


# In[22]:


df_fr = df[df['COUNTRY'] == 1]


# In[38]:



# Calculez la matrice de corrélation
corr_matrix = df_fr.corr()

# Créez une masque pour la diagonale
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Définissez la taille de la figure
plt.figure(figsize=(10, 8))

# Affichez la matrice de corrélation avec des couleurs originales
sns.heatmap(corr_matrix, mask=mask, cmap="Greens", vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink":.5},
            annot=False, fmt=".2f", annot_kws={"size": 6})

# Ajoutez un titre
plt.title("Matrice de corrélation diagonale")

# Affichez la figure
plt.show()


# In[39]:


X = df_fr.drop(columns=['ID', 'TARGET', 'COUNTRY'])
y = df_fr['TARGET']


Sep_Train_Evaluation(X,y,pipe_l)
Sep_Train_Evaluation(X,y,pipe_rf)
Sep_Train_Evaluation(X,y,pipe_gb)


# Here the best model is the bagging model RandomForest and we want to have the best model so we can play on this hyperparameters
# 
# - n_estimators: Number of trees the algorithm builds before averaging the predictions.
# - max_features: Maximum number of features random forest considers splitting a node.
# - mini_sample_leaf: Determines the minimum number of leaves required to split an internal node.
# - Criterion: How to split the node in each tree? (Entropy/Gini impurity/Log Loss)
# - max_leaf_nodes: Maximum leaf nodes in each tree

# In[40]:


rf1_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
rf2_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

pipe_rf1 = Pipeline(std_scaler,rf1_model)
pipe_rf2 = Pipeline(std_scaler,rf2_model)

Sep_Train_Evaluation(X,y,pipe_rf1)
Sep_Train_Evaluation(X,y,pipe_rf2)


# ## 6. Model choice with pipeline construction (DE Dataset)

# In[41]:


df_de = df[df['COUNTRY'] == 0]


# In[43]:


# Calculez la matrice de corrélation
corr_matrix = df_de.corr()

# Créez une masque pour la diagonale
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Définissez la taille de la figure
plt.figure(figsize=(10, 8))

# Affichez la matrice de corrélation avec des couleurs originales
sns.heatmap(corr_matrix, mask=mask, cmap="magma", vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink":.5},
            annot=False, fmt=".2f", annot_kws={"size": 6})

# Ajoutez un titre
plt.title("Matrice de corrélation diagonale")

# Affichez la figure
plt.show()


# In[44]:


X = df_de.drop(columns=['ID', 'TARGET', 'COUNTRY'])
y = df_de['TARGET']

Sep_Train_Evaluation(X,y,pipe_l)
Sep_Train_Evaluation(X,y,pipe_rf)
Sep_Train_Evaluation(X,y,pipe_gb)


# ## 7. Cross Validation

# In[45]:


# Fonction de validation croisée pour un modèle de régression en utilisant la corrélation de Spearman
def kfoldCrossValidation(X,y,M,k):
  # Set up k-fold cross-validation
  kfold = KFold(n_splits=k, shuffle=True, random_state=42)
  scores = []
  X = np.array(X)
  y = np.array(y)
  # Perform k-fold cross-validation
  for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Fit the classifier on the training data
    M.fit(X_train, y_train)
    # Predict on the test data
    y_pred_test = M.predict(X_test)
    # Calculate accuracy and store in scores list
    spearman = spearmanr(y_test, y_pred_test).correlation
    scores.append(spearman)
  return scores


# In[46]:


X = df_fr.drop(columns=['ID', 'TARGET', 'COUNTRY'])
y = df_fr['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled =std_scaler.fit_transform(X_train)

scores = kfoldCrossValidation(X_train_scaled, y_train, rf1_model, k=3)
for fold, acc in enumerate(scores, 1):
    print(f"Fold {fold} accuracy: {acc}")
average_accuracy = sum(scores) / len(scores)
print(f"Average accuracy: {average_accuracy}")

scores = kfoldCrossValidation(X_train_scaled, y_train, rf2_model, k=3)
for fold, acc in enumerate(scores, 1):
    print(f"Fold {fold} accuracy: {acc}")
average_accuracy = sum(scores) / len(scores)
print(f"Average accuracy: {average_accuracy}")


# In[47]:


X = df_de.drop(columns=['ID', 'TARGET', 'COUNTRY'])
y = df_de['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled =std_scaler.fit_transform(X_train)

scores=kfoldCrossValidation(X_train_scaled, y_train,l_model,3)
for fold, acc in enumerate(scores, 1):
    print(f"Fold {fold} accuracy: {acc}")
average_accuracy = sum(scores) / len(scores)
print(f"Average accuracy: {average_accuracy}")


# # Annexe/Test

# ## 8. Use of PCA

# In[30]:


df_fr = df[df['COUNTRY'] == 1]
X = df_fr.drop(columns=['ID', 'TARGET', 'COUNTRY'])
y = df_fr['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n = 15
pca = PCA(n_components=n)
X_pca_train = pca.fit_transform(X_train)
X_pca_test = pca.transform(X_test)

explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(1, n+1), explained_variance)
plt.xlabel('nombre de composante')
plt.ylabel('variance cumulative expliquée')
plt.grid(True)
plt.show()


# In[31]:


rf_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
rf_model.fit(X_pca_train, y_train)

y_pred = rf_model.predict(X_pca_test)
y_pred_train= rf_model.predict(X_pca_train)

print(f"Spearman score: {spearmanr(y_test,y_pred).correlation}")


# In[32]:


df_de = df[df['COUNTRY'] == 0]
X = df_de.drop(columns=['ID', 'TARGET', 'COUNTRY'])
y = df_de['TARGET']

pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Calculer le score Spearman
print(f"Spearman score: {spearmanr(y_test,y_pred).correlation}")


# In[ ]:




