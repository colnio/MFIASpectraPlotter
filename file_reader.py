import pandas as pd
import numpy as np

def read_data(data_file, header_file):
    header_df = pd.read_csv(header_file, sep=';')
    df=pd.read_csv(data_file, sep=';')
    df = df.T
    df.columns = df.iloc[3]
    df.drop(['chunk', 'timestamp', 'fieldname', 'size'], inplace=True)
    return df, header_df

def get_sample_names(header_df):
    return header_df.history_name.unique()

def get_field_names(data_df):
    return list(set(data_df.columns))

def extract_sample_field(data_df, header_df, sample_name, field_name):
    idx = header_df.history_name.str.contains(sample_name)
    dd = data_df[field_name].values.T.astype(float)
    return dd[idx]