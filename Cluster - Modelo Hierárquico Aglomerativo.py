# MBA em Data Science e Analytics USP ESALQ

#Análise de Cluster - Método Hierárquico Aglomerativo - para identificação de acessos massivos ao Aplicativo Menor Preço RS

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

#%% Gráfico 3D das observações

fig = px.scatter_3d(
    robos_dados,
    x='qtd_consultas_no_periodo',
    y='qtd_dias_com_media',
    z='media_dif_minutos_periodo',
    color=robos_dados['qtd_consultas_no_periodo'] > 1000  # True / False
)
fig.show()


#%% Cluster hierárquico aglomerativo: distância euclidiana + single linkage

##Complete Linkage deu os melhores resultados para fins de método de encadeamento.Ainda assim os dendogramas ficaram sem interpratabilidade.
## Distancia euclidiana não deram bons resultados, vamos tentar com a Distância de Manhattan (City Block)
# Visualizando as distâncias

dist_cityblock = pdist(robos_pad, metric='cityblock')

# Opções para as distâncias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

# Gerando o dendrograma
## Distância euclidiana e método de encadeamento single linkage

plt.figure(figsize=(16,8))
dend_sing = sch.linkage(robos_pad, method = 'complete', metric = 'cityblock')
dendrogram_s = sch.dendrogram(dend_sing)
plt.title('Dendrograma Complete Linkage', fontsize=16)
plt.xlabel('CPFs', fontsize=16)
plt.ylabel('Distância Cityblock', fontsize=16)
plt.show()

# Opções para o método de encadeamento ("method"):
    ## single
    ## complete
    ## average

# Opções para as distâncias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

# Gerando a variável com a indicação do cluster no dataset

## Deve ser realizada a seguinte parametrização:
    ## Número de clusters (n_clusters)
    ## Medida de distância (metric)
    ## Método de encadeamento (linkage)
    
# Como já observamos 3 clusters no dendrograma, vamos selecionar "3" clusters
# A medida de distância e o método de encadeamento são mantidos

cluster_complete = AgglomerativeClustering(n_clusters = 3, metric = 'cityblock', linkage = 'complete')
indica_cluster_comp = cluster_complete.fit_predict(robos_pad)
robos_dados_originais['cluster_complete'] = indica_cluster_comp
robos_pad ['cluster_complete'] = indica_cluster_comp
robos_dados_originais['cluster_complete'] = robos_dados_originais['cluster_complete'].astype('category')
robos_pad['cluster_complete'] = robos_pad['cluster_complete'].astype('category')

# Coeficientes do esquema hierárquico de aglomeração (single)
coef_single = [y[1] for y in dendrogram_s['dcoord']]
print(coef_single)

#%% Análise de variância de um fator (ANOVA)

# Interpretação do output:

## cluster_kmeans MS: indica a variabilidade entre grupos
## Within MS: indica a variabilidade dentro dos grupos
## F: estatística de teste (cluster_kmeans MS / Within MS)
## p-unc: p-valor da estatística F
## se p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

# Diferença de tempo médio em minutos entre acessos no mesmo dia
pg.anova(dv='media_dif_minutos_periodo', 
         between='cluster_complete', 
         data=robos_pad,
         detailed=True).T

# # Quantidade de dias com média, ou seja, quantidade de dias com dois ou mais acessos
pg.anova(dv='qtd_dias_com_media', 
         between='cluster_complete', 
         data=robos_pad,
         detailed=True).T

# Quantidade de consultas no período
pg.anova(dv='qtd_consultas_no_periodo', 
         between='cluster_complete', 
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
                    color='cluster_complete')
fig.show()


#%% Identificação das características dos clusters

# Agrupando o banco de dados

analise_robos = robos_dados_originais.drop(columns=['cod_cpf']).groupby(by=['cluster_complete'])

# Estatísticas descritivas por grupo

tab_medias_grupo = analise_robos.mean().T
tab_desc_grupo = analise_robos.describe().T

#%% FIM

