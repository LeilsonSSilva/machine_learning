#Importar Bibliotecas

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv

#Importar dados (dataset disponível no Google Colab) - /sample_data/california_housing_train.csv

dados= pd.read_csv('california_housing_train.csv')
#print(dados.head(10))



#Análise Exploratória

# Id - Id única de cada local
# latitude- latitude do local
# longitude- longitude do local
# median_age- mediana das idades das casas no local
# total_rooms - contagem do total de cômodos na região
# total_bedrooms - contagem do total de quartos na região
# population - população total na região
# households - número total de casas na região
# median_income- mediana da renda das pessoas na região (em dezenas de milhares por ano)
# median_house_value -mediana dos valores das casas da região - variável-alvo
# Este conjunto de dados contém uma linha por grupo de blocos do censo. Um grupo de quarteirões é a menor unidade geográfica para a qual o U.S. Census Bureau publica dados de amostra (um grupo de quarteirões normalmente tem uma população de 600 a 3.000 pessoas).


#print(dados.describe(include='all'))


analise=sv.analyze(dados)
#analise.show_html()


#Mapa de calor (correlação de variáveis)

mask = np.triu(np.ones_like(dados.corr(), dtype=np.bool))
plt.figure(figsize=(10,8))
sns.heatmap(dados.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1)
plt.show()

