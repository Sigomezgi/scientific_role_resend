import pandas as pd

def get_categorical_variables()-> list:
    """get categorical data

    Returns:
        list: categorical variables
    """
    return ["SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

def get_continuos_variables()->list:
    """get numerical data

    Returns:
        list: numerical variables
    """
    return ["LIMIT_BAL","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

def get_numerical_variables()-> list:
    """get numerical variables

    Returns:
        list: list of numerical data
    """
    categorical_variables=get_categorical_variables()
    continuos_variables=get_continuos_variables()
    return categorical_variables+continuos_variables

def cast_numerical_data(data_credit: pd.DataFrame) -> pd.DataFrame:

    """cast numerical data to guarantee format

    Args:
        data_to_cast (pd.DataFrame)
        variables (list): variables to  cast data
        

    Returns:
        pd.DataFrame: Casted Data
    """
    variables= get_numerical_variables()
    data_credit= data_credit.copy()
    data_credit[variables] = pd.to_numeric(data_credit[variables].stack(), errors="coerce").unstack()
    
    return data_credit

def fill_empty_cells(data_credit: pd.DataFrame)-> pd.DataFrame:
    """fill nan's values with an specific metric

    Args:
        data_credit (pd.DataFrame): credit card's database

    Returns:
        pd.DataFrame: data without nan's
    """

    data_credit= data_credit.copy()
    continuos_variables= get_continuos_variables()
    categorical_variables= get_categorical_variables()

    data_credit[continuos_variables]= data_credit[continuos_variables].fillna(data_credit[continuos_variables].median())
    data_credit[categorical_variables]= data_credit[categorical_variables].fillna(data_credit[categorical_variables].mode())
    return data_credit 

def get_clean_data()-> pd.DataFrame:
    """clean dirty data

    Returns:
        pd.DataFrame: Clean data
    """

    credit_data= pd.read_csv("./data/default_of_credit_card_clients.csv", sep=",")
    casted_data= cast_numerical_data(credit_data)
    cleaned_data= fill_empty_cells(casted_data)
    return cleaned_data












