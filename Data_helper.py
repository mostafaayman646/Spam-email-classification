import pandas as pd

def Load_data(path = 'Data/spam.csv'):
    return pd.read_csv(path,encoding='latin-1')

def Save_data(df):
    return df.to_csv(r"Data/spam_cleaned.csv", index=False, encoding='utf-8')