# mini-curso-machine-learning

# Dataset
Foram Feitos 4 teste com 2 modificações do dataset 

1 - dataset com todas as features (é tirado apenas as features que tem as labels nulas)

2 - dataset onde eu tiro todas as features que estao com valores nulos 

3 - Nas direções cardinais onde o dataset informa a direção do vento eu dei um numero para cada direção.Com a logica que cada numero representa um numero e a junção das letras representaria apenas a concatenação dos numero

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

Com todas as features consegui uma acuracia de 86% em todos os testes efetuados e na que eu tirei alguns features deu o mesmo resultado 86% com uma diferença muito baixa entre elas. Fiz dezenas de teste e essa foi a melhor resultado que eu obtive.
