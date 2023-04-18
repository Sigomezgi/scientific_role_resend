import pandas as pd
from sklearn.utils import resample
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as jb
from module_train.fixture_engineer import get_data_with_fixture

def get_sample_data(data:pd.DataFrame, rows:int)-> pd.DataFrame:
    """Sample of data

    Args:
        data (pd.DataFrame): Data to get data sample
        rows (int): Quantity of rows

    Returns:
        pd.DataFrame: Sampled data
    """
    data= data.copy()
    return data.sample(rows)

def get_continuos_variables()->list:
    """get continuos variables of model

    Returns:
        list: Continuos variables of model
    """
    return ["LIMIT_BAL","BILL_MEDIAN","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

def delete_outlier(data_to_elimnate_outlier: pd.DataFrame)->pd.DataFrame:
    """Delete outliers using z score's values

    Args:
        data_to_elimnate_outlier (pd.DataFrame): Data to delete

    Returns:
        pd.DataFrame: Data without outlier
    """
    data_to_elimnate_outlier= data_to_elimnate_outlier.copy()
    continuos_variables= get_continuos_variables()
    #Identify and delete outlier
    data_to_elimnate_outlier = data_to_elimnate_outlier[(np.abs(stats.zscore(data_to_elimnate_outlier[continuos_variables])) < 3).all(axis=1)]

    return data_to_elimnate_outlier

def transform_continuos_variables(credit_data: pd.DataFrame)->pd.DataFrame:
    """Transform continuos data

    Args:
        credit_data (pd.DataFrame): Credit data

    Returns:
        pd.DataFrame: Data with numerical variables transformed
    """
    credit_data= credit_data.copy()
    credit_data_without_outlier_to_transformed= delete_outlier(credit_data)
    columns_to_transform= get_continuos_variables()

    std_scaler= StandardScaler()
    credit_data_without_outlier_to_transformed[columns_to_transform]= std_scaler.fit_transform(credit_data_without_outlier_to_transformed[columns_to_transform])

    jb.dump(std_scaler, "objects/trasforms"+".pkl")
    credit_data_without_outlier_to_transformed.reset_index(inplace=True, drop=True)
    
    return credit_data_without_outlier_to_transformed

def balance_subsampling_majority(data_to_balance: pd.DataFrame)-> pd.DataFrame:
    """Balance train data set

    Args:
        data_to_balance (pd.DataFrame): Train data

    Returns:
        pd.DataFrame: Balanced train data
    """
    data_to_balance= data_to_balance.copy()
    data_variables = data_to_balance.drop("default payment next month", axis=1)
    target = data_to_balance["default payment next month"]
    data_variables_class_0 = data_variables[target == 0]
    data_variables_class_1 = data_variables[target == 1]

    data_variables_class_0_downsampled = resample(data_variables_class_0,
                                 replace=False,  # no se reemplazan las observaciones
                                 n_samples=len(data_variables_class_1))  # número de observaciones de la clase minoritaria
    data_balanced = pd.concat([data_variables_class_0_downsampled, data_variables_class_1])
    target_balanced= np.concatenate([np.zeros(len(data_variables_class_0_downsampled)), np.ones(len(data_variables_class_1))])
    
    data_balanced["default payment next month"] = target_balanced

    return data_balanced

def transform_test_data(data_to_test: pd.DataFrame)-> pd.DataFrame:
    """Transform test data

    Args:
        data_to_test (pd.DataFrame): Test data

    Returns:
        pd.DataFrame: Test data transformed
    """

    data_to_test= data_to_test.copy()
    std_sca: StandardScaler= jb.load("objects/trasforms.pkl")
    continuos_variables= get_continuos_variables()

    data_to_test[continuos_variables]= std_sca.transform(data_to_test[continuos_variables])
    return data_to_test

    

def preprocess_data():

    credit_data_with_fixture= get_data_with_fixture()
    credit_data_train= get_sample_data(credit_data_with_fixture, 10000)
    credit_data_test= get_sample_data(credit_data_with_fixture, 500)
    credit_data_train_transformed= transform_continuos_variables(credit_data_train)
    credit_data_train_balanced= balance_subsampling_majority(credit_data_train_transformed)
    credit_data_test_transformed= transform_test_data(credit_data_test)

    return credit_data_train_balanced, credit_data_test_transformed
