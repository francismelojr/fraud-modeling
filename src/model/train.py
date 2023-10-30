# pipeline
# salvar o modelo
import pickle

import cloudpickle
import numpy as np

# manipulação de dados
import pandas as pd
from category_encoders import WOEEncoder

# feature engineering
from feature_engine import encoding, imputation
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import DropFeatures

# modelos de machine learning
from lightgbm import LGBMClassifier

# métricas
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve

# pré-processamento
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# tuning de hiperparâmetros
from skopt import BayesSearchCV

# arquivo a ser salvo

output_file = 'src/model/model_lgbm.bin'

# importing data
data = pd.read_csv('data/data.csv')

# data splitting
X = data.drop(columns=['fraude'])
y = data['fraude']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.2, stratify=y
)

# pre processing
replace_N = ['entrega_doc_2']

imputer_N = imputation.CategoricalImputer(fill_value='N', variables=replace_N)

replace_median = [
    'score_2',
    'score_3',
    'score_4',
    'score_6',
    'score_9',
    'score_10',
]

imputer_median = imputation.MeanMedianImputer(
    imputation_method='median', variables=replace_median
)

replace_x = ['pais']

imputer_x = imputation.CategoricalImputer(fill_value='XX', variables=replace_x)

# Função para criar colunas de indicadores de missing values

missing_cols = [
    'entrega_doc_2',
    'score_3',
    'score_2',
    'score_4',
    'score_10',
    'pais',
    'score_6',
    'score_9',
]


def create_missing_indicator_columns(X):
    missing_indicator_columns = pd.DataFrame()
    for column in missing_cols:
        missing_indicator = X[column].isnull().astype(int)
        missing_indicator_columns[f'is_missing_{column}'] = missing_indicator

    return pd.concat([X, missing_indicator_columns], axis=1)


missing_indicator_transformer = FunctionTransformer(
    create_missing_indicator_columns, validate=False
)

# feature engineering

# WOE encoder


class WOEEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.encoder = WOEEncoder(cols=cols)

    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        X_encoded = self.encoder.transform(X)
        return X_encoded


# binary encoder


class BinaryLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column] = X_copy[column].map({'Y': 1, 'N': 0})
        return X_copy


columns_to_encode = ['entrega_doc_2', 'entrega_doc_3']

label_encoder = BinaryLabelEncoder(columns=columns_to_encode)

# date hour extractor
date_variable = ['data_compra']

hour_extractor = DatetimeFeatures(
    variables=date_variable, features_to_extract=['hour']
)

# remove columns
columns_to_remove = ['produto']

drop_features = DropFeatures(features_to_drop=columns_to_remove)

# model training
ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
model = LGBMClassifier(scale_pos_weight=ratio, random_state=0, verbose=-1)

space_lgbm = {
    'learning_rate': (0.035, 0.065),
    'num_leaves': (20, 40),
    'min_child_samples': (20, 40),
    'subsample': (0.25, 0.55),
    'colsample_bytree': (0.35, 0.65),
    'max_depth': (12, 16),
}

bayes_search = BayesSearchCV(
    model,
    space_lgbm,
    cv=5,
    scoring='roc_auc',
    n_iter=30,
    verbose=0,
    random_state=0,
)

pipeline = Pipeline(
    [
        ('missing_indicator', missing_indicator_transformer),
        ('imputer_N', imputer_N),
        ('imputer_median', imputer_median),
        ('imputer_x', imputer_x),
        (
            'WOE Encoder',
            WOEEncoderTransformer(cols=['categoria_produto', 'pais']),
        ),
        ('label_encoder', label_encoder),
        ('hour_extractor', hour_extractor),
        ('drop_features', drop_features),
        ('model', bayes_search),
    ]
)

# treinando o modelo
pipeline.fit(X_train, y_train)

# predict e scores
y_train_pred = pipeline.predict(X_train)
y_train_score = pipeline.predict_proba(X_train)[:, 1]

y_test_pred = pipeline.predict(X_test)
y_test_score = pipeline.predict_proba(X_test)[:, 1]

# função para achar melhor threshold
def best_threshold(X, y_score):
    valor_compra = X['valor_compra'].values
    thresholds = np.linspace(0, 1, num=100)

    data = []
    profits = []

    best_threshold = 0
    max_value = 0
    for threshold in thresholds:
        true_negatives = (y_score < threshold) & (y_train == 0)
        false_negatives = (y_score < threshold) & (y_train == 1)

        revenue = np.sum(0.1 * (true_negatives * valor_compra))
        loss = np.sum(false_negatives * valor_compra)

        profit = revenue - loss
        profits.append(profit)
        if profit > max_value:
            max_value = profit
            best_threshold = threshold
        data.append([threshold, revenue, loss, profit])
    return best_threshold


# achando o melhor threshold para o conjunto de treino

best_threshold = best_threshold(X_train, y_train_score)

# métricas de treino

train_precision = metrics.precision_score(
    y_train, y_train_score > best_threshold
)
train_recall = metrics.recall_score(y_train, y_train_score > best_threshold)
train_f1_score = metrics.f1_score(y_train, y_train_score > best_threshold)
train_auc = metrics.roc_auc_score(y_train, y_train_score)

# métricas de teste

test_precision = metrics.precision_score(y_test, y_test_score > best_threshold)
test_recall = metrics.recall_score(y_test, y_test_score > best_threshold)
test_f1_score = metrics.f1_score(y_test, y_test_score > best_threshold)
test_auc = metrics.roc_auc_score(y_test, y_test_score)

print(f'Melhor Threshold: {best_threshold}\n')

print('Métricas de Treino\n--------')
print(f'AUC: {train_auc}')
print(f'Precision: {train_precision}')
print(f'Recall: {train_recall}')
print(f'F1-Score: {train_f1_score}\n')

print('Métricas de Teste\n--------')
print(f'AUC: {test_auc}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1-Score: {test_f1_score}\n')

with open(output_file, 'wb') as f_out:
    cloudpickle.dump((pipeline, best_threshold), f_out)
