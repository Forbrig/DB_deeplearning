"""
O conjunto de dados a ser utilizado (em anexo) apresenta dados de pessoas que
procuraram serviço hospitalar por um determinado motivo. O conjunto de dados
possui 49 atributos e um rótulo. O rótulo possui três valores distintos:
o paciente não retornou ao hospital (NO), retornou em 30 dias (<30)
ou retornou após 30 dias (>30).
O link https://www.hindawi.com/journals/bmri/2014/781670/tab1/ possui uma
tabela com a descrição dos atributos e rótulo (eliminado o /tab1/ do link você
terá acesso ao artigo que descreve experimentos com o conjunto de dados).
Alguns atributos são alfanuméricos e devem ser discretizados. Por exemplo
"age" está organizado por faixa de valores [0-10), [10,20), e assim por diante.
O conjunto de dados possui alguns atributos com valores faltantes. Os mesmos
deverão ser tratados. Por exemplo o atributo "weigth" possui quase 99.000
valores faltantes.

Passos:
1. Identifique os 10 melhores atributos para a criação do modelo
(por exemplo, utilizando o SelectKBest).
2. Selecione 3 algoritmos classificadores para a atividade.
3. Define um conjuno de hiper parâmetros para os modelos
(selecione pelo menos 3 hiper parâmetros) e defina pelo menos 2
valores para cada hiper parâmetro.
4. Utilizando o GridSearchCV encontre a melhor combinação de parâmetros
para os 3 classificadores (utilize 5 fatias para a validação cruzada).
5. Rode os classificadores com um novo conjunto (subconjunto do original)
ainda não utilizado (cerca de 10%).
6. Utilize o método classification_report do pacote metrics para apresentar
a performance dos classificadores.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier


from names import names


def preprocessing_values(df):
    """
    Realiza a etapa de pré-processamento do dataset
    :param df: dataset
    :return: clean dataset
    """

    # troca os valores ? por NaN, tipo nulo do numpy
    df = df.replace('?', np.NaN)
    # ds.isnull().sum() : mostra quais sãos os valores nulos no dataset.

    # Essa parte remove os atributos `weight` pela falta e valores e
    # `payer_code` por não ser relevante na classificação.
    df.drop(['weight', 'payer_code'], inplace=True, axis=1)

    # no atributo `medical_specialty` é incluido o valor "missing"
    # para os valores nulos.
    df['medical_specialty'].fillna('missing', inplace=True)

    # Para os valores nulos de `race`, é atribuido o valor mais recorrente.
    df['race'].fillna(df['race'].mode()[0], inplace=True)

    # Para os valores nulos de `diag_1`, é atribuido o valor mais recorrente.
    df['diag_1'].fillna(df['diag_1'].mode()[0], inplace=True)

    # Para os valores nulos de `diag_2`, é atribuido o valor mais recorrente.
    df['diag_2'].fillna(df['diag_2'].mode()[0], inplace=True)

    # Para os valores nulos de `diag_3`, é atribuido o valor mais recorrente.
    df['diag_3'].fillna(df['diag_3'].mode()[0], inplace=True)

    # valores categoricos são transformados em numéricos
    numeric_attrs = [n[0] for n in names if n[1] == 'numeric']
    nominal_attrs = [n[0] for n in names if n[1] == 'nominal']

    df_num = df[numeric_attrs]
    df_nom = df[nominal_attrs]
    df_nom = df_nom.apply(LabelEncoder().fit_transform)

    df = pd.concat([df_num, df_nom], axis=1)

    # conforme descrito no artigo, é utilizado somente uma ocorrência de
    # internação para cada paciente, no caso é utilizado a primeira ocorrência
    df = df.drop_duplicates('patient_nbr', keep='first')

    return df


# carrega o dataset disponibilizado no moodle
dataset = pd.read_csv('diabetic_data.csv', sep=',')
df = preprocessing_values(dataset)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# 1. Identifique os 10 melhores atributos para a criação do modelo
X = SelectKBest(k=10).fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

# 2. Selecione 3 algoritmos classificadores para a atividade.
ET = ExtraTreeClassifier()
DT = DecisionTreeClassifier()
KNN = KNeighborsClassifier()
clf_list = [ET, DT, KNN]

# 3. Define um conjuno de hiper parâmetros para os modelos
# (selecione pelo menos 3 hiper parâmetros) e defina pelo menos 2
# valores para cada hiper parâmetro.
extra_hparams = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 3]
}
tree_hparams = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 3]
}
knn_hparams = {
    'n_neighbors': [15, 20],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree']
}
hparams_list = [extra_hparams, tree_hparams, knn_hparams]

# 4. Utilizando o GridSearchCV encontre a melhor combinação de parâmetros
# para os 3 classificadores (utilize 5 fatias para a validação cruzada).

for clf, hparams in zip(clf_list, hparams_list):
    grid = GridSearchCV(clf, hparams, cv=5)
    print('Running... {}'.format(clf.__class__.__name__))
    model = grid.fit(X_train, y_train)

    print('Best hyper params:')
    print(model.best_estimator_)

    # 5. Rode os classificadores com um novo conjunto (subconjunto do original)
    # ainda não utilizado (cerca de 10%).
    y_pred = model.predict(X_test)
    print('Results:')
    print(classification_report(y_test, y_pred))
