#Importar Bibliotecas

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv




#Importar dados (dataset disponível no Google Colab) - /sample_data/california_housing_train.csv

dados= pd.read_csv('california_housing_train.csv')
#print(dados.head(10))



# print(dados.describe(include='all'))

# analise=sv.analyze(dados)
# analise.show_html()



#Iniciar o processo de Machine Learning

#determinar o alvo (preço médio das casas)

y=dados['median_house_value']
#pode escrever assim também: y=dados.median_house_value
#print(y)



#Escolher as features (váriveis a serem utilizadas do seu conjunto de dados)

features=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
X=dados[features]
#print(X)



#Construindo o modelo

# Usar Scikit-Learn
# Define = Escolha de modelo (define parâmetros)
# Fit = Treinar
# Predict = Fazer a Predição
# Evaluate = Avaliar os resultados

from sklearn.neighbors import KNeighborsRegressor

modelo = KNeighborsRegressor(7)



#Treinar o Modelo

modelo.fit(X,y)
dados.head(5)
# print(X.head(5))



# Fazer a predição

modelo.predict(X.head(5))
# print(modelo.predict(X.head(5)))



# Validar o modelo
# erro=atual - previsao (achar a diferença)

from sklearn.metrics import mean_absolute_error

predicao=modelo.predict(X)
# print(predicao)

mean_absolute_error(y,predicao)
# print(mean_absolute_error(y,predicao))



# Vamos ver isso em um dataframe (valor esperado e predição)

dados2=pd.DataFrame(y)
dados2['predicao']=predicao
#print(dados2.sample(10))

# print(dados2.describe())



# Validação (agora de uma maneira diferente)

from sklearn.model_selection import train_test_split

# Pegar parte dos dados (aqui 80%) pra treinar e ai testar nos outros 20%

treino_X, val_X, treino_y, val_y = train_test_split(X,y,random_state=1,train_size=0.8)
treino_X.shape

# print(treino_X.shape)


modelo2 = KNeighborsRegressor(3)
modelo2.fit(treino_X,treino_y)

predicao2=modelo2.predict(val_X)


# Analisar o erro (ver outros números de vizinhos)

mean_absolute_error(val_y,predicao2)
print(mean_absolute_error(val_y,predicao2))



