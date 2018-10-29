#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 23:40:21 2018

@author: igorpedromartins
"""

import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classes = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split

#   Criando variáveis de treino de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
        previsores, classes, test_size = 0.25)

import keras
#   Sequencial pois o neurônio segue uma sequência na rede neural
from keras.models import Sequential
#   Dense significa que todos os neurônios estão conectados, por exemplo se temos 5 entradas e 
#   na camada oculta temos 10 neurônios as 5 entradas estará ligada a cada um dos neurôrios subsequentes
from keras.layers import Dense

#   Criando nossa rede neural 
classificacao = Sequential()

#   Calculando quantos neurônios será feitos na camada oculta
#   (numeroEntrada + numeroSaida) / 2
#   (30 + 1) / 2
#   Camada de entrada
classificacao.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', 
                        input_dim = 30))

#   Criando mais uma camada oculta, neste caso retiramos o INPUT_DIM que é utilizado 
#   apenas no inicio para entrada de novos dados
classificacao.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))

#   Camada de saida
classificacao.add(Dense(units = 1, activation = 'sigmoid'))

#   Compile aqui vamos fazer a atualização dos pesos ou a descida de gradiente
#   optimizer = atualização de pesos
#   loss = calculo de perda
#   metrics = realiza a metrica de acertos e erros
'''
    classificacao.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                          metrics = ['binary_accuracy'])
'''
otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
classificacao.compile(otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

#   Treinando o algoritmo
classificacao.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)

'''
    Mostrando pesos encontrados dentro da rede neural
    Por padrão quando criado mais um neurônio que chamamos de BIAS(viés) este neurônio auxilia na 
    hora de classificar algum problema, sendo assim ajuda a cometer menos erros de classificação.
'''
peso0 = classificacao.layers[0].get_weights()
print(peso0)
print(len(peso0))
peso1 = classificacao.layers[1].get_weights()
peso2 = classificacao.layers[2].get_weights()

#   Fazendo previsões
previsoes = classificacao.predict(previsores_teste)
#   Tratando como 0 e 1, mas neste caso será seta True or False
previsoes = (previsoes > 0.5)

#   Verificando a porcentagem de acerto do algortimo
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matrix_confusao = confusion_matrix(classe_teste, previsoes)


