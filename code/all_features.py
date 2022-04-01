import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from functions import organizar_dados , holdout, validacao_cruzada

print("Pegando features do dataset")

train = pd.read_csv(os.path.join('..','Dataset','input','train.csv'))
test =  pd.read_csv(os.path.join('..','Dataset','input','test.csv'))

#------------------------------------------------------------------------------#

''' Pre-processamento'''


all_features = True
vc = False

SEED = 9305
val_porcentagem = 0.20

print("Organizando dados...")

features, teste , labels_treino, labels_teste = organizar_dados(vc ,train, test, all_features)


#------------------------------------------------------------------------------#

'''Treinamento
    Sera feito 2 teste nesse arquivo primeiro todos com os 2 modelos
    Teste_1 com o hold 
    Teste_2 com a validação Cruzada
'''
nome_arquivo = "all_features"

import xgboost as xgb
modelo_1 = xgb.XGBClassifier()
modelo_2 = SVC()

print("#--------------------------------------------------------#")
# #TESTE_1
print("Teste_1 Holdout")
print("Modelo = XGBClassifier")
holdout(modelo_1,features, teste,labels_treino,labels_teste, nome_arquivo)
print('Modelo = SVC')
holdout(modelo_2,features, teste,labels_treino,labels_teste, nome_arquivo)

print("#--------------------------------------------------------#")
#TESTE_2

X = features
y = labels_treino

n_splits = 5
kFold = StratifiedKFold(n_splits=n_splits)
kFold.get_n_splits(X, y)

X = np.array(X)
y = np.array(y)
print("Teste_1 Validacao cruzada")
print("Modelo = XGBClassifier")
validacao_cruzada(modelo_1,kFold,X,y,nome_arquivo)
print('Modelo = SVC')
validacao_cruzada(modelo_2,kFold,X,y,nome_arquivo)

#------------------------------------------------------------------------------#