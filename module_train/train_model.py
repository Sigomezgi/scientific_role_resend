import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from module_train.preprocess_data_train import preprocess_data
from sklearn.metrics import recall_score

def make_grid_search_cv(train_data: pd.DataFrame, test_data: pd.DataFrame)-> list:
    """Evaluate models

    Args:
        train_data (pd.DataFrame): train data
        test_data (pd.DataFrame): test_data

    Returns:
        list: metrics of  grid
    """

    train_data= train_data.copy()
    test_data= test_data.copy()

    variables = list (train_data.columns)
    target= "default payment next month"
    variables.remove(target)
    variables.remove("ID")

    svc= SVC()
    rfc = RandomForestClassifier()
    gbc= GradientBoostingClassifier()
    lgb= LGBMClassifier()
    all_models =[svc, rfc, gbc, lgb]

    params_SVC = get_params_svc()
    params_rfc= get_params_rf()
    params_gbc= get_params_gbc()
    params_lgb= get_params_lgb()
    all_params= [params_SVC, params_rfc, params_gbc, params_lgb]

    names = ["svc","rfc","gbc","lgb"]
    best_params = {}
    score_test= {}
    score_evalua= {}

    for i,j,k in zip(all_models, all_params, names):
        grid = GridSearchCV(estimator=i, param_grid=j, scoring='accuracy', cv=3)
        grid.fit(train_data[variables], train_data[target])
        predict_evalua= grid.predict(test_data[variables])

        best_params[k] = grid.best_estimator_
        score_test[k] = grid.best_score_
        score_evalua[k] = recall_score(test_data[target].values, predict_evalua)

    results = [best_params, score_test, score_evalua]

    return results
    

def get_params_svc()-> dict:
    """get paramns svc

    Returns:
        dict: Params svc
    """
    param_grid = {'kernel': ['linear', 'rbf'], 
              'C': [0.1, 1, 10],
              'gamma': [0.01, 0.1, 1]}
    return param_grid

def get_params_rf()-> dict:
    """get params random forest

    Returns:
        dict: Random forest param
    """
    param_grid = {'n_estimators': [100, 200, 500],
              'max_depth': [10, 20, 30, None],
              'max_features': ['sqrt', 'log2']}
    
    return param_grid

def get_params_gbc()-> dict:
    """get params gbc

    Returns:
        dict: gbc params
    """
    param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]}
    return param_grid

def get_params_lgb()-> dict:
    """get params light gradient boosting

    Returns:
        dict: lgb params
    """
    param_grid = {
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]}

    return param_grid



def train_model():

    credit_data_train_balanced, credit_data_test_transformed = preprocess_data()
    results_data= make_grid_search_cv(credit_data_train_balanced, credit_data_test_transformed)

    return results_data