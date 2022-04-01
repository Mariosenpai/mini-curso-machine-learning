import os
import math
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.svm import SVC


def ler_dataset(a, all_features = True):
    a = a[a['RainTomorrow'].notna()] # tira as linhas nulas que tem labels nulas
    if all_features == False:
        a = a[a['MinTemp'].notna()] # tira as linhas nulas que tem labels nulas
        a = a[a['MaxTemp'].notna()]
        a = a[a['Evaporation'].notna()]
        a = a[a['Sunshine'].notna()]
        a = a[a['WindGustDir'].notna()]
        a = a[a['WindGustSpeed'].notna()]
        a = a[a['WindDir9am'].notna()]
        a = a[a['WindDir3pm'].notna()]
        a = a[a['WindSpeed9am'].notna()]
        a = a[a['WindSpeed3pm'].notna()]
        a = a[a['WindSpeed9am'].notna()]
        a = a[a['WindSpeed3pm'].notna()]
        a = a[a['Humidity9am'].notna()]
        a = a[a['Humidity3pm'].notna()]
        a = a[a['Pressure9am'].notna()]
        a = a[a['Pressure3pm'].notna()]
        a = a[a['Cloud9am'].notna()]
        a = a[a['Cloud3pm'].notna()]
        a = a[a['Temp9am'].notna()]
        a = a[a['Temp3pm'].notna()]
        a = a[a['RainToday'].notna()]

    #tratando todas os nan (nulos) e Direções da tabela
    a = a.fillna(-1)
    a = a.mask(a == 'W' , 4)
    a = a.mask(a == 'WNW' , 484)
    a = a.mask(a == 'WSW' , 424)
    a = a.mask(a == 'NE' , 86)
    a = a.mask(a == 'NNW' , 884)
    a = a.mask(a == 'N' , 8)
    a = a.mask(a == 'NNE' , 886)
    a = a.mask(a == 'SW' , 28)
    a = a.mask(a == 'ENE' , 686)
    a = a.mask(a == 'SSE' , 226)
    a = a.mask(a == 'S' , 2)
    a = a.mask(a == 'NW' , 84)
    a = a.mask(a == 'SE' , 26)
    a = a.mask(a == 'ESE' , 626)
    a = a.mask(a == 'E' , 6)
    a = a.mask(a == 'SSW' , 228)

    a = a.mask(a == 'Yes' , 1)
    a = a.mask(a == 'No' , 0)

    features = a.drop(columns = ['Id', 'Date' , 'Location', 'RainTomorrow'])
    labels = a.get('RainTomorrow')

    return features , labels

def teste(model):
    modelo = model

    print('Treinando...')
    modelo.fit(X,y)
    previsoes = modelo.predict(val_treino)
    acc = accuracy_score(labels_val, previsoes)
    acuracia = acc  * 100
    matriz_confusao = sklearn.metrics.confusion_matrix(labels_val, previsoes)

    print('Matriz de confusao\n',matriz_confusao)
    print(acuracia)

    print('')

def evaluate(params):
    print('Parametros usados ', params)
    modelo = SVC(kernel = params['kernel'],random_state = 42
               ,gamma =params['gamma'])

    print('Treinando...')
    modelo.fit(X,y)
    previsoes = modelo.predict(val_treino)
    acc = accuracy_score(labels_val, previsoes)
    acuracia = acc  * 100
    matriz_confusao = sklearn.metrics.confusion_matrix(labels_val, previsoes)

    print('Matriz de confusao',matriz_confusao)
    print(acuracia)

    print('')
    
    return {'loss': 1-acc, 'status': STATUS_OK}

def organizar_dados(validacao_cruzada, train , test):
    scaler = StandardScaler()
    teste = 0
    labels_teste = 0
    if validacao_cruzada :
        features , labels = ler_dataset(train)
        test_features , test_labels = ler_dataset(test)

        features = features + test_features
        labels = labels + test_labels
        
        #normalizar
        scaler.fit(features)
        features = scaler.transform(features)
        labels_treino = labels.astype('int')
    else:
        features , labels = ler_dataset(train)
        test_features , test_labels = ler_dataset(test)
        
        #normalizar
        scaler.fit(features)
        features = scaler.transform(features)
        teste = scaler.transform(test_features)
        labels_treino = labels.astype('int')
        labels_teste = test_labels.astype('int')
    return features , teste, labels_treino, labels_teste   
    
def validacao_cruzada(modelo,X,y, nome_arquivo):
    cont= 0
    acc = []
    for train_index, test_index in kFold.split(X,y):
        print(f"Validacao cruzada, foram {cont+1} de {n_splits}")

        #----------------------------------------------------------------------------#
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #----------------------------------------------------------------------------#

        modelo.fit(X_train,y_train)
        previsoes = modelo.predict(X_test)
        acuracia = accuracy_score(y_test, previsoes)
        matriz_confusao = sklearn.metrics.confusion_matrix(y_test, previsoes)

        # print('Matriz de confusao',matriz_confusao)
        print('Acuracia : ',acuracia)
        acc.append(acuracia)

        f1 = sklearn.metrics.f1_score(y_test, previsoes)
        precision = sklearn.metrics.precision_score(y_test, previsoes)
        recall = sklearn.metrics.recall_score(y_test, previsoes)
        #------------------------------------------------------------------------------#
        '''salva o log'''
        
        
        info = 'Validacao Cruzada - ' +str(cont) + ' de 5\n'
        classificacao_geral = classification_report(y_test, previsoes)
        print(classificacao_geral,'\n')
        print('matriz de confusao:')
        mc = sklearn.metrics.confusion_matrix(y_test, previsoes)
        print(mc)
        linha = '\n#-----------------------------------------------------#\n'
        log = linha + info + classificacao_geral
        log = log +'\nMatriz de confussao:\n' + str(mc)
        log = log + '\n\n\nAcuracia: ' + str(acuracia)  
        log = log + '\n\nF1_Score: ' + str(f1) 
        log = log + '\n\nPrecision: ' + str(precision)
        log = log + '\n\nRecall: ' + str(recall)


        nome_log = nome_arquivo+'_Valicacao_Cruzada_'+str(modelo)+'.txt'
        caminho_log = os.path.join('Testes', nome_arquivo , nome_log)
        with open(caminho_log , 'a') as arquivo:
            arquivo.write(log);
        cont=cont + 1

    best_acc = acc[0]
    for i in acc:
        if i > best_acc:
            best_acc = i 
    print('---------------------------------')
    log = '\n'+linha+'\nMelhor acc : '+ str(best_acc)+'\n'+linha
    with open(caminho_log , 'a') as arquivo:
        arquivo.write(log);

    return best_acc

def holdout(modelo, x_train,x_test,y_train, y_test,nome_arquivo):    
    acc =[]
    #----------------------------------------------------------------------------#

    modelo.fit(X_train,y_train)
    previsoes = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, previsoes)
    matriz_confusao = sklearn.metrics.confusion_matrix(y_test, previsoes)

    # print('Matriz de confusao',matriz_confusao)
    print('Acuracia : ',acuracia)
    acc.append(acuracia)

    f1 = sklearn.metrics.f1_score(y_test, previsoes)
    precision = sklearn.metrics.precision_score(y_test, previsoes)
    recall = sklearn.metrics.recall_score(y_test, previsoes)
    #------------------------------------------------------------------------------#
    '''salva o log'''

    info = 'Holdout\n'
    classificacao_geral = classification_report(y_test, previsoes)
    print(classificacao_geral,'\n')
    print('matriz de confusao:')
    mc = sklearn.metrics.confusion_matrix(y_test, previsoes)
    print(mc)
    linha = '\n#-----------------------------------------------------#\n'
    log = linha + info + classificacao_geral
    log = log +'\nMatriz de confussao:\n' + str(mc)
    log = log + '\n\n\nAcuracia: ' + str(acuracia)  
    log = log + '\n\nF1_Score: ' + str(f1) 
    log = log + '\n\nPrecision: ' + str(precision)
    log = log + '\n\nRecall: ' + str(recall)

    nome_log = nome_arquivo+'_Holdout_'+str(modelo)+'.txt'
    caminho_log = os.path.join('Testes', nome_arquivo , nome_log)
    with open(caminho_log , 'a') as arquivo:
        arquivo.write(log);

    best_acc = acc[0]
    for i in acc:
        if i > best_acc:
            best_acc = i 
    print('---------------------------------')
    log = '\n'+linha+'\nMelhor acc : '+ str(best_acc)+'\n'+linha
    with open(caminho_log , 'a') as arquivo:
        arquivo.write(log);