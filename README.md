# mini-curso-machine-learning

# Dataset
Foram Feitos 4 teste com 2 modificações do dataset 

1 - dataset com todas as features (é tirado apenas as features que tem as labels nulas)

2 - dataset onde eu tiro todas as features que estao com valores nulos 

3 - Nas direções cardeais onde o dataset informa a direção do vento, foi dado um numero para cada direção. Com a logica de que cada numero representa um numero e a junção das letras representaria apenas a concatenação dos numero.

Exemplo 'N' = 8 e 'W' = 4 entao 'NW' = 48, 
ficando assim
 'W' = 4,
 'WNW' = 484,
 'WSW' = 424,
    'NE' = 86,
    'NNW' = 884,
    'N' = 8,
    'NNE' = 886,
   'SW' = 28,
    'ENE' = 686,
    'SSE' = 226,
    'S' = 2,
    'NW' = 84,
    'SE' = 26,
    'ESE' = 626,
    'E' = 6,
'SSW' = 228.
# Modelo

Foram usados 2 Modelos para o treinamento

1 - SVC 

Encontrado em https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svc#sklearn.svm.SVC

Foi rodado um otimizado no modelo para determinado o Kernel e o gamma. Conclui que o Kernel "poly" e o gamma "scale" eram os melhores para o modelo

2 - xgboost

Encontrado em https://xgboost.readthedocs.io/en/stable/python/python_api.html

# Testes

Os testes efetivados com todas as features conseguiu alcança uma acuracia de 86%(a mais alto de todos os teste feitos) e na que foi tirada alguns features, deu o mesmo resultado 86%, com uma diferença muito baixa entre elas. Com isso concluimos que o tirar aquelas features que estavam com valores nulos não influenciou muito no resultado. Fiz dezenas de teste e essa foi a melhor resultado que eu obtive.
