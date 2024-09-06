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
from sklearn.utils.fixes import loguniform
from scipy.stats import randint as sp_randint
import warnings
warnings.filterwarnings("ignore")


param1 = {
        "randomforestclassifier__max_depth": sp_randint(3, 20),
        "randomforestclassifier__n_estimators": sp_randint(50, 200),
        "randomforestclassifier__min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),   
        "randomforestclassifier__min_samples_split": sp_randint(2, 10),
        "randomforestclassifier__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "randomforestclassifier__bootstrap": [True, False]
    }
param2 = {
        "decisiontreeclassifier__max_depth": sp_randint(3, 20),
        "decisiontreeclassifier__min_samples_leaf": sp_randint(1, 10),   
        "decisiontreeclassifier__min_samples_split": sp_randint(2, 10),
        "decisiontreeclassifier__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "decisiontreeclassifier__criterion": ['gini', 'entropy']
    }
param3 = {
    "xgbclassifier__learning_rate": uniform(0.01, 0.29),
    "xgbclassifier__n_estimators": randint(10, 100),
    "xgbclassifier__max_depth": randint(2, 8),
    "xgbclassifier__min_child_weight": randint(1, 6),
    "xgbclassifier__subsample": uniform(0.5, 0.5),
    "xgbclassifier__colsample_bytree": uniform(0.5, 0.5),
    "xgbclassifier__gamma": uniform(0, 5)
    }
param4 = {
        "histgradientboostingclassifier__max_iter": sp_randint(50, 200),
        "histgradientboostingclassifier__min_samples_leaf": sp_randint(10, 30),
        "histgradientboostingclassifier__max_bins":randint(2, 255),
        "histgradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3], 
        "histgradientboostingclassifier__max_depth": sp_randint(3, 10)
        #"l2_regularization": random.uniform(0.0, 100.0)
    }
param5 = {
        "gradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "gradientboostingclassifier__max_depth": sp_randint(3, 10),
        "gradientboostingclassifier__n_estimators": sp_randint(50, 200),
        "gradientboostingclassifier__min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),   
        "gradientboostingclassifier__min_samples_split": sp_randint(2, 10),
        "gradientboostingclassifier__max_features": ['sqrt', 'log2', 0.5, 0.8]
    }
param6 = {
        "extratreesclassifier__max_depth": sp_randint(3, 20),
        "extratreesclassifier__n_estimators": sp_randint(50, 200), 
        "extratreesclassifier__min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),   
        "extratreesclassifier__min_samples_split": sp_randint(2, 10),
        "extratreesclassifier__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "extratreesclassifier__bootstrap": [True, False]
    }
param7 = {
        
        "adaboostclassifier__n_estimators": sp_randint(30, 200), 
        "adaboostclassifier__learning_rate": [0.01, 0.1, 0.5, 1],   
        #'base_estimator': DecisionTreeClassifier(min_samples_split= 1.0, min_samples_leaf= 0.30000000000000004, max_features= 13, max_depth= 14.0, criterion= 'entropy')
    }
param8 = {
        
        'svc__C': uniform(0.1, 9.9),
        'svc__gamma': ['scale', 'auto'] + list(uniform(0.001, 0.999).rvs(10)),
        'svc__kernel': ['rbf','poly','sigmoid','linear'],
        'svc__degree':[1,2,3,4,5,6]
        
    }
param9 = {
        
        'logisticregression__C': uniform(0.1, 9.9),
        'logisticregression__penalty': ['l1', 'l2'],
        'logisticregression__solver': ['liblinear', 'saga'],
        'logisticregression__max_iter' : [100, 1000, 2500, 5000]
        
    }
param10 = {
        
        'lineardiscriminantanalysis__solver': ['lsqr', 'eigen'],
        'lineardiscriminantanalysis__shrinkage': [None, 'auto', 0.1, 0.5, 0.9]      
    }
param11 = {
        
        'kneighborsclassifier__n_neighbors': randint(3, 20),
        'kneighborsclassifier__weights': [ 'distance'],
        'kneighborsclassifier__metric': ['euclidean']     
    }
param12 = {
        
      # 'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
       'gaussiannb__var_smoothing': np.logspace(0,-9, num=100)    
    }

n_splits = 5  # Nombre de plis pour la validation croisée
n_iter = 500  # Nombre d'itérations pour RandomizedSearchCV
param_model = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12]
pipe1 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), RandomForestClassifier(random_state = 693))
pipe2 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), DecisionTreeClassifier(random_state = 693))
pipe3 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), XGBClassifier(random_state = 693))
pipe4 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), HistGradientBoostingClassifier(random_state = 693))
pipe5 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), GradientBoostingClassifier(random_state = 693))
pipe6 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), ExtraTreesClassifier(random_state = 693))
pipe7 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), AdaBoostClassifier(random_state = 693))
pipe8 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), SVC(random_state = 693))
pipe9 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), LogisticRegression(random_state = 693))
pipe10 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), LinearDiscriminantAnalysis())
pipe11 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), KNeighborsClassifier())
pipe12 = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), GaussianNB())

pipes = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, pipe7, pipe8, pipe9, pipe10, pipe11, pipe12]
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

print('----------------------------------------LASSO--------------------------------------------------')
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()

pipeline = Pipeline([
                    
                    ('model',LogisticRegression(penalty='l1',solver='liblinear',random_state=693, max_iter=10000, class_weight='balanced'))])
 
from sklearn.metrics import r2_score
 
search = GridSearchCV(pipeline,
                    {'model__C':np.logspace(-6, 1, 100)},
                    cv = 5, scoring='roc_auc',verbose=1
                    )
 
search.fit(X_train,np.ravel(Y_train))
 
print(search.best_params_)
 
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
print('imp', len(importance[0]))
features=list(X_train.columns)
selectd=np.array(features)[importance[0] > 0]
print('Lasso', selectd)
 
X_train=X_train[selectd].copy()
X_validation=X_validation[selectd].copy()

best_scores = []
validation_scores = []
for pipe, model, params in zip(pipes, models, param_model):
    clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = n_iter, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train, np.ravel(Y_train))

    best_score = clf.best_score_
    best_params = clf.best_params_
    # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': best_score})
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, Meilleur score AUC: {best_score}")

    # Création du pipeline final
    final_model = type(model)(**{k.split("__")[1]: v for k, v in best_params.items() if k.startswith(type(model).__name__.lower())})
    final_pipe = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), final_model)
    final_pipe.fit(X_train, np.ravel(Y_train))

    y_pred_validation = final_pipe.predict(X_validation)
    final_validation_score = roc_auc_score(np.ravel(Y_validation), y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score}")
# Convertir les listes en DataFrames
df_best_scoresLasso = pd.DataFrame(best_scores)
df_best_scoresLasso.set_index(df_best_scoresLasso.columns[0], inplace=True)
df_best_scoresLasso.rename(columns={'Best Score': 'LASSO'}, inplace=True)
df_validation_scoresLasso = pd.DataFrame(validation_scores)
df_validation_scoresLasso.set_index(df_validation_scoresLasso.columns[0], inplace=True)
df_validation_scoresLasso.rename(columns={'Validation Score': 'LASSO'}, inplace=True)

print('------------------------------------------------------------------------------------------')
print('----------------------------------------LSVC--------------------------------------------------')
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()

pipeline = Pipeline([
                    
                    ('model',LinearSVC(penalty="l1", dual=False, random_state = 693, class_weight = 'balanced', max_iter = 10000))])
 
from sklearn.metrics import r2_score
 
search = GridSearchCV(pipeline,
                    {'model__C':np.logspace(-6, 1, 100)},
                    cv = 5, scoring='roc_auc',verbose=1
                    )
 
search.fit(X_train,np.ravel(Y_train))
print("Meilleur score : ", search.best_score_)
print(search.best_params_)
 
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
print('imp', len(importance[0]))
features=list(X_train.columns)
selectd=np.array(features)[importance[0] > 0]
print('Lsvc', selectd)
X_train=X_train[selectd].copy()
X_validation=X_validation[selectd].copy()

best_scores = []
validation_scores = []
for pipe, model, params in zip(pipes, models, param_model):
    clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = n_iter, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train, np.ravel(Y_train))

    best_score = clf.best_score_
    best_params = clf.best_params_
    # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': best_score})
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, Meilleur score AUC: {best_score}")

    # Création du pipeline final
    final_model = type(model)(**{k.split("__")[1]: v for k, v in best_params.items() if k.startswith(type(model).__name__.lower())})
    final_pipe = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), final_model)
    final_pipe.fit(X_train, np.ravel(Y_train))

    y_pred_validation = final_pipe.predict(X_validation)
    final_validation_score = roc_auc_score(np.ravel(Y_validation), y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score}")
# Convertir les listes en DataFrames
df_best_scoreslsvc = pd.DataFrame(best_scores)
df_best_scoreslsvc.set_index(df_best_scoreslsvc.columns[0], inplace=True)
df_best_scoreslsvc.rename(columns={'Best Score': 'LSVC'}, inplace=True)
df_validation_scoreslsvc = pd.DataFrame(validation_scores)
df_validation_scoreslsvc.set_index(df_validation_scoreslsvc.columns[0], inplace=True)
df_validation_scoreslsvc.rename(columns={'Validation Score': 'LSVC'}, inplace=True)
print('------------------------------------------------------------------------------------------')
print('----------------------------------------RFECV--------------------------------------------------')
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()

#estimator = RandomForestClassifier(random_state =693)
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X_train, Y_train)
feature_idx = selector.get_support()
feature_name = X_train.columns[feature_idx]
X_train = X_train[feature_name]
X_validation = X_validation[feature_name]
print('RFECV', X_train.columns)

best_scores = []
validation_scores = []
for pipe, model, params in zip(pipes, models, param_model):
    clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = n_iter, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train, np.ravel(Y_train))

    best_score = clf.best_score_
    best_params = clf.best_params_
    # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': best_score})
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, Meilleur score AUC: {best_score}")

    # Création du pipeline final
    final_model = type(model)(**{k.split("__")[1]: v for k, v in best_params.items() if k.startswith(type(model).__name__.lower())})
    final_pipe = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), final_model)
    final_pipe.fit(X_train, np.ravel(Y_train))

    y_pred_validation = final_pipe.predict(X_validation)
    final_validation_score = roc_auc_score(np.ravel(Y_validation), y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score}")
# Convertir les listes en DataFrames
df_best_scoresrfecv = pd.DataFrame(best_scores)
df_best_scoresrfecv.set_index(df_best_scoresrfecv.columns[0], inplace=True)
df_best_scoresrfecv.rename(columns={'Best Score': 'RFECV'}, inplace=True)
df_validation_scoresrfecv = pd.DataFrame(validation_scores)
df_validation_scoresrfecv.set_index(df_validation_scoresrfecv.columns[0], inplace=True)
df_validation_scoresrfecv.rename(columns={'Validation Score': 'RFECV'}, inplace=True)
print('------------------------------------------------------------------------------------------')
print('----------------------------------------TREE BASED FEATURE--------------------------------------------------')
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
params = {
        "max_depth": np.linspace(1, 32, 32, endpoint=True),
        "n_estimators": [1, 2, 4, 8, 16, 32, 64, 100, 200], 
        "min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),   
        "min_samples_split": np.linspace(0.1, 1.0, 10, endpoint=True),
        "max_features": list(range(1,X_train.shape[1])),
        "bootstrap": [True, False]
    }

random_search = RandomizedSearchCV(estimator=ExtraTreesClassifier(random_state = 693), random_state = 693, param_distributions=params, n_iter=100, cv=5, verbose=1, n_jobs=-1)
    # Entraînement de RandomizedSearchCV
random_search.fit(X_train, np.ravel(Y_train))

model = ExtraTreesClassifier(**random_search.best_params_ , random_state = 693).fit(X_train, np.ravel(Y_train))

selector = SelectFromModel(model, prefit=True)
feature_idx = selector.get_support()
feature_name = X_train.columns[feature_idx]
X_train = X_train[feature_name]
X_validation = X_validation[feature_name]
print('TBF', X_train.columns)
best_scores = []
validation_scores = []
for pipe, model, params in zip(pipes, models, param_model):
    clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = n_iter, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train, np.ravel(Y_train))

    best_score = clf.best_score_
    best_params = clf.best_params_
    # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': best_score})
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, Meilleur score AUC: {best_score}")

    # Création du pipeline final
    final_model = type(model)(**{k.split("__")[1]: v for k, v in best_params.items() if k.startswith(type(model).__name__.lower())})
    final_pipe = make_pipeline(SVMSMOTE(sampling_strategy='minority', random_state=42), final_model)
    final_pipe.fit(X_train, np.ravel(Y_train))

    y_pred_validation = final_pipe.predict(X_validation)
    final_validation_score = roc_auc_score(np.ravel(Y_validation), y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score}")
# Convertir les listes en DataFrames
df_best_scorestbf = pd.DataFrame(best_scores)
df_best_scorestbf.set_index(df_best_scorestbf.columns[0], inplace=True)
df_best_scorestbf.rename(columns={'Best Score': 'TBF'}, inplace=True)
df_validation_scorestbf = pd.DataFrame(validation_scores)
df_validation_scorestbf.set_index(df_validation_scorestbf.columns[0], inplace=True)
df_validation_scorestbf.rename(columns={'Validation Score': 'TBF'}, inplace=True)
print('------------------------------------------------------------------------------------------')

##DOWNLOAD DATA
auc_cv = pd.concat([df_best_scoresLasso, df_best_scoreslsvc, df_best_scoresrfecv, df_best_scorestbf], axis=1)
auc_valid = pd.concat([df_validation_scoresLasso, df_validation_scoreslsvc, df_validation_scoresrfecv, df_validation_scorestbf], axis=1)
print(auc_cv)
print(auc_valid)
auc_cv.to_excel('/home/vad3/Project 2/donnee model/modelewsilevel/PDL1/path/AUC_CV_1.xlsx', index=False)
auc_valid.to_excel('/home/vad3/Project 2/donnee model/modelewsilevel/PDL1/path/AUC_VALID_1.xlsx', index=False)
auc_cv.to_csv('/home/vad3/Project 2/donnee model/modelewsilevel/PDL1/path/AUC_CV_1.csv', index=False)
auc_valid.to_csv('/home/vad3/Project 2/donnee model/modelewsilevel/PDL1/path/AUC_VALID_1.csv', index=False)