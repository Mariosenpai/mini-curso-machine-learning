import os
import math
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from functions import ler_dataset, teste , evaluate, validacao_cruzada

a = pd.read_csv(os.path.join('dataset','input','database.csv'))

train = pd.read_csv(os.path.join('train.csv'))
test =  pd.read_csv(os.path.join('test.csv'))

features , labels = ler_dataset(train)
test_features , test_labels = ler_dataset(test)

''' Pre-processamento'''


validacao_cruzada = False

SEED = 9305
val_porcentagem = 0.20

scaler = StandardScaler()

if validacao_cruzada :
  features , labels = ler_dataset(train)
  test_features , test_labels = ler_dataset(test)

  features = features + test_features
  labels = labels + test_labels

  scaler.fit(features)
  features = scaler.transform(features)
  labels_treino = labels.astype('int')
else:
  features , labels = ler_dataset(train)
  test_features , test_labels = ler_dataset(test)

  scaler.fit(features)
  features = scaler.transform(features)
  teste = scaler.transform(test_features)
  labels_treino = labels.astype('int')
  labels_teste = test_labels.astype('int')


#Normalizar 
scaler = StandardScaler()
scaler.fit(data_treino)
data_treino = scaler.transform(data_treino)
val_treino = scaler.transform(val_treino)

labels_treino = labels_treino.astype('int')
labels_val = labels_val.astype('int')


'''Treinamento'''


X = data_treino
y = labels_treino

n_splits = 5
kFold = StratifiedKFold(n_splits=n_splits)
kFold.get_n_splits(X, y)

X = np.array(X)
y = np.array(y)

import xgboost as xgb
modelo_1 = xgb.XGBClassifier()
modelo_2 = SVC()

validacao_cruzada(modelo_1)
validacao_cruzada(modelo_2)