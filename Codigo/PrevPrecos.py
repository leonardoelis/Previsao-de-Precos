import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

housing = pd.read_csv('housing.csv')
print("Cabeçalho Arquivo")
print(housing.head())

# 10 features, 20.639 amostras, "total_bedrooms" possui valores faltantes
print("Info Arquivo")
print(housing.info())

# Única feature não-numérica
print("Feature Não Numérica")
print(housing["ocean_proximity"].value_counts())

print(housing.describe())

# Histograma de todas as features
housing.hist(bins=50, figsize=(20, 15))
plt.show()

# Correlação entre as features
cor = housing.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(cor, annot=True)
plt.show()
# Na matriz de correlação gerada, pode-se observar que a feature "median_income"
# possui a maior correlação com a variável target ("median_house_value"), então essa feature será analisada

# Histograma da feature "median_income"
housing["median_income"].hist()
plt.show()
# No histograma plotado, pode-se notar que a maioria dos valores estão entre 2 e 5, mas alguns valores vão muito além de 6
# Por isso será usada uma amostragem estratificada para dividir os conjuntos de treino e teste, fazendo com que os conjuntos
# tenham valores representativos de cada uma das categorias de "median_income"

# Criação da feature "income_cat" que contém os valores de "median_income" dispostos em 5 categorias
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 2., 4., 6., 8., np.inf], labels=[0, 1, 2, 3, 4])

# Conversão da coluna "income_cat" de category para numérico
cat_columns = housing.select_dtypes(['category']).columns
housing[cat_columns] = housing[cat_columns].apply(lambda x: x.cat.codes)

# Histograma de "income_cat"
housing["income_cat"].hist()
plt.show()

# Divisão em conjuntos de treino e teste
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Eliminar o target dos dados de treino e copiar o target para as labels
train_set = strat_train_set.drop("median_house_value", axis=1)
labels = strat_train_set["median_house_value"].copy()
print(train_set.info())

test_set = strat_test_set.drop("median_house_value", axis=1)
test_labels = strat_test_set["median_house_value"].copy()

# Remoção da feature não numérica
housing_num = train_set.drop('ocean_proximity', axis=1)

# pipeline para pré-processamento das variáveis numéricas (imputação de valores faltantes com a mediana e normalização dos dados)
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

housing_num_tr = num_transformer.fit(housing_num)

# pipeline para pré-processamento das variáveis categóricas (substituição dos valores da variável categórica por valores numéricos)
cat_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder())
])

# Compondo os pré-processadores
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                              'households', 'median_income', 'income_cat']),
    ('cat', cat_transformer, ['ocean_proximity'])
])

# Criando o modelo com pipeline para Regressão Linear
model_lin_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lin_reg', LinearRegression())
])

# Utilização do GridSearch para a escolha da melhor combinação de parâmetros da LinearRegression()
parameters = {'lin_reg__fit_intercept': [True, False],
              'lin_reg__normalize': [True, False]}
grid = GridSearchCV(model_lin_reg, parameters, cv=5, verbose=100)
grid.fit(train_set, labels)
print("================================RESULTADOS REGRESSÃO LINEAR================================")
print(grid.best_params_)
print(grid.best_score_)
print(grid.score(test_set, test_labels))

# Criando o modelo com pipeline para Random Forest Regressor
model_random_forest = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('random_forest', RandomForestRegressor())
])

# Utilização do GridSearch para a escolha da melhor combinação de parâmetros da RandomForestRegressor()
parameters = {'random_forest__n_estimators': [10, 50, 90, 100],
              'random_forest__max_features': ["auto", "sqrt", "log2"],
              'random_forest__bootstrap': [True, False]}
grid = GridSearchCV(model_random_forest, parameters, cv=5, verbose=100)
grid.fit(train_set, labels)
print("================================RESULTADOS RANDOM FOREST================================")
print(grid.best_params_)
print(grid.best_score_)
print(grid.score(test_set, test_labels))
