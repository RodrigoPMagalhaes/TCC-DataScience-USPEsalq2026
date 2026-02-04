# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:00:47 2025

@author: rpm_r
"""# MBA em Data Science e Analytics USP ESALQ

#Análise de Cluster - Método Não Hierárquico - Kmeans - para identificação de acessos massivos ao Aplicativo Menor Preço RS

#%% Instalando os pacotes


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

#%% Importando o banco de dados

robos_dados_originais = pd.read_csv('Cluster.csv')


#%% Visualizando informações sobre os dados e variáveis

# Estrutura do banco de dados

print(robos_dados_originais.info())

# Vamos remover a coluna "cod_cpf", pois trata-se de id
# robos_dados é o banco original sem a ID (cod_cpf)

robos_dados = robos_dados_originais.drop(columns=['cod_cpf'])

# Estatísticas descritivas das variáveis

tab_desc = robos_dados.describe()

# Gerando a matriz de correlações de Pearson
matriz_corr = pg.rcorr(robos_dados, method = 'pearson', upper = 'pval', 
                       decimals = 4, 
                       pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

print(matriz_corr)


#%% Mapa de calor indicando a correlação entre os atributos

# Matriz de correlações básica
corr = robos_dados.corr()

# Gráfico de calor (heatmap)
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = corr.columns,
        y = corr.index,
        z = np.array(corr),
        text=corr.values,
        texttemplate='%{text:.2f}',
        colorscale='viridis'))

fig.update_layout(
    height = 600,
    width = 600)

fig.show()
#%% Padronização das variáveis

# Aplicando o procedimento de ZScore em todas as variáveis - Excluir ID - robos_dados e não robos_dados_originais
robos_pad = robos_dados.apply(zscore, ddof=1)

print(round(robos_pad.mean(), 2))
print(round(robos_pad.std(), 2))
# As variáveis passam a ter média = 0 e desvio padrão = 1

#%% Gráfico 3D das observações

fig = px.scatter_3d(robos_dados, 
                    x='qtd_consultas_no_periodo', 
                    y='media_dif_minutos_periodo', 
                    z='qtd_dias_com_media')
fig.show()

#%% Gráfico 3D das observações padronizadas

fig = px.scatter_3d(robos_pad, 
                    x='qtd_consultas_no_periodo', 
                    y='qtd_dias_com_media', 
                    z='media_dif_minutos_periodo')
fig.show()

## Gráfico com Cores

fig = px.scatter_3d(
    robos_dados,
    x='qtd_consultas_no_periodo',
    y='qtd_dias_com_media',
    z='media_dif_minutos_periodo',
    color=robos_dados['qtd_consultas_no_periodo'] > 1000  # True / False
)
fig.show()



#%% Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1,11) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(robos_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11)) # ajustar range
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

#%% Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2,11) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(robos_pad)
    silhueta.append(silhouette_score(robos_pad, kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 11), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()

#%% Cluster Não Hierárquico K-means

# Vamos considerar 3 clusters, considerando as evidências anteriores!

kmeans_final = KMeans(n_clusters = 3, init = 'random', random_state=100).fit(robos_pad)

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans_final.labels_
robos_dados_originais['cluster_kmeans'] = kmeans_clusters
robos_pad['cluster_kmeans'] = kmeans_clusters
robos_dados_originais['cluster_kmeans'] = robos_dados_originais['cluster_kmeans'].astype('category')
robos_pad['cluster_kmeans'] = robos_pad['cluster_kmeans'].astype('category')

#%% Análise de variância de um fator (ANOVA)

# Interpretação do output:

## cluster_kmeans MS: indica a variabilidade entre grupos
## Within MS: indica a variabilidade dentro dos grupos
## F: estatística de teste (cluster_kmeans MS / Within MS)
## p-unc: p-valor da estatística F
## se p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais


# Diferença de tempo médio em minutos entre acessos no mesmo dia
pg.anova(dv='media_dif_minutos_periodo', 
         between='cluster_kmeans', 
         data=robos_pad,
         detailed=True).T

# # Quantidade de dias com média, ou seja, quantidade de dias com dois ou mais acessos
pg.anova(dv='qtd_dias_com_media', 
         between='cluster_kmeans', 
         data=robos_pad,
         detailed=True).T

# Quantidade de consultas no período
pg.anova(dv='qtd_consultas_no_periodo', 
         between='cluster_kmeans', 
         data= robos_pad,
         detailed=True).T

## A variável mais discriminante contém a maior estatística F (e significativa)
## O valor da estatística F é sensível ao tamanho da amostra

#%% Gráfico 3D dos clusters

# Perspectiva 1


fig = px.scatter_3d(robos_dados_originais,  
                    x='qtd_consultas_no_periodo',
                    y='qtd_dias_com_media',
                    z='media_dif_minutos_periodo',
                    color='cluster_kmeans')

fig.show()

# Perspectiva 2
fig = px.scatter_3d(robos_dados_originais,  
                    x='qtd_consultas_no_periodo',
                    y='media_dif_minutos_periodo',
                    z='qtd_dias_com_media',
                    color='cluster_kmeans')

fig.show()

# Dados padronizados
fig = px.scatter_3d(robos_pad,  
                    x='qtd_consultas_no_periodo',
                    y='qtd_dias_com_media',
                    z='media_dif_minutos_periodo',
                    color='cluster_kmeans')

fig.show()


#%% Identificação das características dos clusters

# Agrupando o banco de dados

cartao_grupo = robos_dados_originais.groupby(by=['cluster_kmeans'])

# Estatísticas descritivas por grupo

tab_desc_grupo = cartao_grupo.describe().T


#%% Quantidade por categoria

contagem = robos_dados_originais['cluster_kmeans'].value_counts().reset_index()
contagem.columns = ['categoria', 'qtde']
print(contagem)


resumo = (
    robos_dados_originais
        .groupby('cluster_kmeans')
        .agg(
            qtde_registros = ('cluster_kmeans', 'size'),          # quantos em cada categoria
            soma_consultas = ('qtd_consultas_no_periodo', 'sum')  # soma das consultas
        )
        .reset_index()
)

print(resumo)
#%% Resumo
 
 Clusters        qtde_registros  soma_consultas
0 Robôs                   2997              15331236
1 Humanos menos ativos    508                 4534
2 Humanos mais ativos     12755              912616


#%%
cpf_robo = robos_dados_originais[['cod_cpf', 'cluster_kmeans']]

#%% Exportar para CSV

cpf_robo.to_csv('cpf_robo.csv', index=False, sep=';')

#%% Fim

