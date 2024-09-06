import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import warnings
from sklearn.decomposition import FactorAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from imblearn.pipeline import Pipeline, make_pipeline
from skrebate import ReliefF
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import VotingClassifier
from skrebate import SURF
from skrebate import SURFstar
from skrebate import MultiSURF
from skrebate import MultiSURFstar
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from scipy.stats import uniform, randint
from scipy.stats import randint as sp_randint
import warnings
warnings.filterwarnings("ignore")

models = [RandomForestClassifier(random_state = 693), DecisionTreeClassifier(random_state = 693), XGBClassifier(random_state = 693), 
HistGradientBoostingClassifier(random_state = 693), GradientBoostingClassifier(random_state = 693), 
          ExtraTreesClassifier(random_state = 693), AdaBoostClassifier(random_state = 693), SVC(random_state = 693), LogisticRegression(random_state = 693),
         LinearDiscriminantAnalysis(), KNeighborsClassifier(), GaussianNB()]

X_validation = pd.read_csv('/home/vad3/Project 2/donnee model/modelewsilevel/PDL1/path/X_validation.csv')
Y_validation = pd.read_csv('/home/vad3/Project 2/donnee model/modelewsilevel/PDL1/path/Y_validation.csv')
X_train = pd.read_csv('/home/vad3/Project 2/donnee model/modelewsilevel/PDL1/path/X_train.csv')
Y_train = pd.read_csv('/home/vad3/Project 2/donnee model/modelewsilevel/PDL1/path/Y_train.csv')
X_train_init = X_train.copy()
X_validation_init = X_validation.copy()

print('----------------------------------------RELIEF--------------------------------------------------')

param1 = {
        "model__max_depth": sp_randint(3, 20),
        "model__n_estimators": sp_randint(50, 200),
        "model__min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),   
        "model__min_samples_split": sp_randint(2, 10),
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__bootstrap": [True, False]
    }
param2 = {
        "model__max_depth": sp_randint(3, 20),
        "model__min_samples_leaf": sp_randint(1, 10),   
        "model__min_samples_split": sp_randint(2, 10),
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__criterion": ['gini', 'entropy']
    }
param3 = {
    'model__learning_rate': uniform(0.01, 0.29),
    'model__n_estimators': randint(10, 100),
    'model__max_depth': randint(2, 8),
    'model__min_child_weight': randint(1, 6),
    'model__subsample': uniform(0.5, 0.5),
    'model__colsample_bytree': uniform(0.5, 0.5),
    'model__gamma': uniform(0, 5)
    }
param4 = {
        "model__max_iter": sp_randint(50, 200),
        "model__min_samples_leaf": sp_randint(10, 30),
        "model__max_bins":randint(2, 255),
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3], 
        "model__max_depth": sp_randint(3, 10)
        #"l2_regularization": random.uniform(0.0, 100.0)
    }
param5 = {
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "model__max_depth": sp_randint(3, 10),
        "model__n_estimators": sp_randint(50, 200),
        "model__min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),   
        "model__min_samples_split": sp_randint(2, 10),
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8]
    }
param6 = {
        "model__max_depth": sp_randint(3, 20),
        "model__n_estimators": sp_randint(50, 200), 
        "model__min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),   
        "model__min_samples_split": sp_randint(2, 10),
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__bootstrap": [True, False]
    }
param7 = {
        
        "model__n_estimators": sp_randint(30, 200), 
        "model__learning_rate": [0.01, 0.1, 0.5, 1],   
        #'base_estimator': DecisionTreeClassifier(min_samples_split= 1.0, min_samples_leaf= 0.30000000000000004, max_features= 13, max_depth= 14.0, criterion= 'entropy')
    }
param8 = {
        
        'model__C': uniform(0.1, 9.9),
        'model__gamma': ['scale', 'auto'] + list(uniform(0.001, 0.999).rvs(10)),
        'model__kernel': ['rbf','poly','sigmoid','linear'],
        'model__degree':[1,2,3,4,5,6]
        
    }
param9 = {
        
        'model__C': uniform(0.1, 9.9),
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga'],
        'model__max_iter' : [100, 1000, 2500, 5000]
        
    }
param10 = {
        
        'model__solver': ['lsqr', 'eigen'],
        'model__shrinkage': [None, 'auto', 0.1, 0.5, 0.9]      
    }
param11 = {
        
        'model__n_neighbors': randint(3, 20),
        'model__weights': [ 'distance'],
        'model__metric': ['euclidean']     
    }
param12 = {
        
      # 'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
       'model__var_smoothing': np.logspace(0,-9, num=100)    
    }
param_model = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12]
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
feature_range = range(3, 54)  # Plage pour le nombre de caractéristiques à sélectionner
n_iter = 100  # Nombre d'itérations pour RandomizedSearchCV

print('----------------------------------------MUTUAL INFORMATION--------------------------------------------------')
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
feature_range = range(3, 54)  # Plage pour le nombre de caractéristiques à sélectionner
n_iter = 100  # Nombre d'itérations pour RandomizedSearchCV
for model, params in zip(models, param_model):
    best_score = 0
    best_params = {'n_features_to_select': None, 'params': None}
    for n_features in feature_range:
        # Créer un pipeline avec ReliefF, SVMSMOTE et le modèle
        pipe = Pipeline([
            ('mi', SelectKBest(mutual_info_classif, k=n_features)),
            ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
            ('model', model)
        ])

        # Configurer RandomizedSearchCV
        clf = RandomizedSearchCV(pipe, params, random_state = 693, n_iter=n_iter, scoring=make_scorer(roc_auc_score), cv=StratifiedKFold(n_splits=n_splits))
        clf.fit(X_train.values, Y_train['pdl1_group'].values)

        if clf.best_score_ > best_score:
            best_score = clf.best_score_
            best_params['n_features_to_select'] = n_features
            best_params['params'] = clf.best_params_
        # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': best_score})
    #print(f"Meilleurs paramètres: {best_params}, Meilleur score AUC: {best_score}")
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, AUC in CV: {best_score}")

    # Préparer les paramètres du modèle
    model_params = {k.split("__")[1]: v for k, v in best_params['params'].items() if k.startswith('model__')}

    # Création du pipeline final
    final_model = type(model)(**model_params)
    final_pipe = Pipeline([
        ('mi', SelectKBest(mutual_info_classif, k=best_params['n_features_to_select'])),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', final_model)])

    final_pipe.fit(X_train.values, np.ravel(Y_train['pdl1_group'].values))

    y_pred_validation = final_pipe.predict(X_validation.values)
    final_validation_score = roc_auc_score(Y_validation['pdl1_group'].values, y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score}")
    #print(f"Score AUC sur l'ensemble de validation: {final_validation_score}")
df_best_scoresmi = pd.DataFrame(best_scores)
df_best_scoresmi.set_index(df_best_scoresmi.columns[0], inplace=True)
df_best_scoresmi.rename(columns={'Best Score': 'MI'}, inplace=True)
df_validation_scoresmi = pd.DataFrame(validation_scores)
df_validation_scoresmi.set_index(df_validation_scoresmi.columns[0], inplace=True)
df_validation_scoresmi.rename(columns={'Validation Score': 'MI'}, inplace=True)
print(df_best_scoresmi)
print(df_validation_scoresmi)
print('------------------------------------------------------------------------------------------')
print('----------------------------------------SURF--------------------------------------------------')
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
feature_range = range(3, 54)  # Plage pour le nombre de caractéristiques à sélectionner
n_iter = 100  # Nombre d'itérations pour RandomizedSearchCV
for model, params in zip(models, param_model):
    best_score = 0
    best_params = {'n_features_to_select': None, 'params': None}
    for n_features in feature_range:
        # Créer un pipeline avec ReliefF, SVMSMOTE et le modèle
        pipe = Pipeline([
            ('surf', SURF( n_features_to_select=n_features)),
            ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
            ('model', model)
        ])

        # Configurer RandomizedSearchCV
        clf = RandomizedSearchCV(pipe, params, random_state = 693, n_iter=n_iter, scoring=make_scorer(roc_auc_score), cv=StratifiedKFold(n_splits=n_splits))
        clf.fit(X_train.values, Y_train['pdl1_group'].values)

        if clf.best_score_ > best_score:
            best_score = clf.best_score_
            best_params['n_features_to_select'] = n_features
            best_params['params'] = clf.best_params_
        # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': best_score})
    #print(f"Meilleurs paramètres: {best_params}, Meilleur score AUC: {best_score}")
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, AUC in CV: {best_score}")

    # Préparer les paramètres du modèle
    model_params = {k.split("__")[1]: v for k, v in best_params['params'].items() if k.startswith('model__')}

    # Création du pipeline final
    final_model = type(model)(**model_params)
    final_pipe = Pipeline([
        ('surf', SURF( n_features_to_select=best_params['n_features_to_select'])),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', final_model)])

    final_pipe.fit(X_train.values, np.ravel(Y_train['pdl1_group'].values))

    y_pred_validation = final_pipe.predict(X_validation.values)
    final_validation_score = roc_auc_score(Y_validation['pdl1_group'].values, y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score}")
    #print(f"Score AUC sur l'ensemble de validation: {final_validation_score}")
df_best_scoressurf = pd.DataFrame(best_scores)
df_best_scoressurf.set_index(df_best_scoressurf.columns[0], inplace=True)
df_best_scoressurf.rename(columns={'Best Score': 'SURF'}, inplace=True)
df_validation_scoressurf = pd.DataFrame(validation_scores)
df_validation_scoressurf.set_index(df_validation_scoressurf.columns[0], inplace=True)
df_validation_scoressurf.rename(columns={'Validation Score': 'SURF'}, inplace=True)
print(df_best_scoressurf)
print(df_validation_scoressurf)
print('------------------------------------------------------------------------------------------')