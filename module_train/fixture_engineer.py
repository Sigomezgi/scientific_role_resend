import pandas as pd
from module_train.clean_data import get_clean_data


def get_billstatement_variables()->list:
    """

    Returns:
        list: Bill statement variables.
    """

    return ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]

def add_median_billstatement(credit_data:pd.DataFrame)->pd.DataFrame:
    """Calculate Bill statement's median

    Returns:
        pd.DataFrame: Data model
    """
    credit_data= credit_data.copy()
    variables_bill_statement = get_billstatement_variables()
    credit_data["BILL_MEDIAN"]= credit_data[variables_bill_statement].median(axis=1)

    return credit_data

def make_variables(credit_data_clean: pd.DataFrame)->pd.DataFrame:
    """Make fixture engineer

    Args:
        credit_data_clean (pd.DataFrame): Credit data cleaned

    Returns:
        pd.DataFrame: Credit data cleaned with new variables
    """
    credit_data_clean= credit_data_clean.copy()
    return add_median_billstatement(credit_data_clean)

def select_columns_to_train(credit_data_with_new_columns:pd.DataFrame)-> pd.DataFrame:
    """Select columns for model training based on the variable analysis. (variable_analysis.ipynb)

    Args:
        credit_data_with_new_columns (pd.DataFrame): Crdit card data with new variables

    Returns:
        pd.DataFrame: credit card data with variables selected
    """
    credit_data_with_new_columns= credit_data_with_new_columns.copy()
    columns_model_train= ['ID', 'LIMIT_BAL', 'SEX','EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6','PAY_2','PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',"BILL_MEDIAN", 'PAY_AMT1','default payment next month']   
    
    return credit_data_with_new_columns[columns_model_train]
    

def get_data_with_fixture()->pd.DataFrame:
    """Make fixture engineer to credit card data

    Returns:
        pd.DataFrame: data make model
    """

    cleaned_data= get_clean_data()
    data_with_new_variables= make_variables(cleaned_data)
    return select_columns_to_train(data_with_new_variables)



    
