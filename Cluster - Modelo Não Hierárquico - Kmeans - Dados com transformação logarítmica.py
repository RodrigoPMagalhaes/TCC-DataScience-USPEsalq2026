# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 01:59:01 2026

@author: rpm_r
"""
#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin
!pip install factor_analyzer
!pip install prince
!pip install statsmodels

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


#%% Importando o banco de dados

dados_originais = pd.read_csv('Cluster.csv')

print(dados_originais.info())

dados_originais_semindex = dados_originais.drop(columns=['cod_cpf'])


cols = ["media_dif_minutos_periodo", "qtd_dias_com_media", "qtd_consultas_no_periodo"]

dados_log = dados_originais_semindex.copy()

# log puro (vai dar -inf se houver 0)
dados_log[cols] = np.log(dados_log[cols].astype(float))

dados_log.head()

#%% Zscore

dados_log_pad = dados_log.copy()

# limpa inf/-inf
dados_log_pad[cols] = dados_log_pad[cols].replace([np.inf, -np.inf], np.nan)

# mantém só linhas válidas para padronizar
mask = dados_log_pad[cols].notna().all(axis=1)

scaler = StandardScaler()
dados_log_pad.loc[mask, cols] = scaler.fit_transform(dados_log_pad.loc[mask, cols].astype(float))

dados_log_pad.head()

#%% Identificação da quantidade de clusters (Método Elbow)

cols = ["media_dif_minutos_periodo", "qtd_dias_com_media", "qtd_consultas_no_periodo"]

X = dados_log_pad[cols].replace([np.inf, -np.inf], np.nan).astype(float).values
X = SimpleImputer(strategy="median").fit_transform(X)

elbow = []
K = range(1, 11)

for k in K:
    km = KMeans(n_clusters=k, init="random", random_state=100, n_init="auto")
    km.fit(X)
    elbow.append(km.inertia_)

plt.figure(figsize=(16,8))
plt.plot(list(K), elbow, marker="o")
plt.xlabel("Nº Clusters", fontsize=16)
plt.xticks(range(1,11))
plt.ylabel("WCSS (inertia)", fontsize=16)
plt.title("Método de Elbow", fontsize=16)
plt.show()

#%% Identificação da quantidade de clusters (Método Silhueta)

sil = []
K = range(2, 11)  # silhueta não existe para k=1

for k in K:
    km = KMeans(n_clusters=k, init="random", random_state=100, n_init="auto")
    labels = km.fit_predict(X)
    sil.append(silhouette_score(X, labels))

plt.figure(figsize=(16, 8))
plt.plot(list(K), sil, marker="o")
plt.xlabel("Nº Clusters", fontsize=16)
plt.xticks(range(2, 11))
plt.ylabel("Silhouette score médio", fontsize=16)
plt.title("Coeficiente de Silhueta", fontsize=16)
plt.show()

print(f"Melhor k (silhueta): {list(K)[int(np.argmax(sil))]} | score: {max(sil):.4f}")

#%% Rodar Clusterização


cols = ["media_dif_minutos_periodo", "qtd_dias_com_media", "qtd_consultas_no_periodo"]

# y sem NaN/inf
mask = dados_log_pad[cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
y = dados_log_pad.loc[mask, cols].astype(float).values

kmeans_final = KMeans(n_clusters=3, init="random", random_state=100, n_init="auto")
labels = kmeans_final.fit_predict(y)

# salva clusters no dataframe (só nas linhas válidas)
dados_log_pad.loc[mask, "cluster"] = labels

# Gerando a variável para identificarmos os clusters gerados

cols = ["media_dif_minutos_periodo", "qtd_dias_com_media", "qtd_consultas_no_periodo"]

# máscara de linhas válidas (sem NaN/inf nas features)
mask = dados_log_pad[cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)

# matriz pra treinar
y = dados_log_pad.loc[mask, cols].astype(float).values

kmeans_final = KMeans(n_clusters=3, init="random", random_state=100, n_init="auto")
labels = kmeans_final.fit_predict(y)

# 1) salvar no próprio dados_log_pad (só linhas válidas)
dados_log_pad["cluster"] = np.nan
dados_log_pad.loc[mask, "cluster"] = labels

# 2) salvar no dados_originais alinhando pelo MESMO index
dados_originais["cluster_kmeans"] = np.nan
dados_originais.loc[dados_log_pad.index[mask], "cluster_kmeans"] = labels



#%% Gerar Gráfico

fig = px.scatter_3d(dados_originais,  
                    x='qtd_consultas_no_periodo',
                    y='qtd_dias_com_media',
                    z='media_dif_minutos_periodo',
                    color='cluster_kmeans')

fig.show()

#%% Estatísticas descritivas do Grupo

cartao_grupo = dados_originais.groupby(by=['cluster_kmeans'])

# Estatísticas descritivas por grupo

tab_desc_grupo = cartao_grupo.describe().T

#%% Exportar para o Excel

dados_originais.to_excel("dados_originais.xlsx", index=True)

#%% Fim