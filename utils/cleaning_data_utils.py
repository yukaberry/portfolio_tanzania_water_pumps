import numpy as np
from sklearn.preprocessing import StandardScaler

def Scaling(df,columns):
    for col in columns:
        scaler= StandardScaler()
        df[col] = scaler.fit_transform(np.array(df[col].values).reshape(-1,1))
    return df[col]


def replace_nan_with_zero(df,variable):
    df_variables=[variable]
    for i in df_variables:
        df[i].replace(0,np.nan,inplace=True)
        return df[i].isnull().sum()

def return_median(df,variable):   
    temp_df = df[df[variable].notnull()]
    temp_df = temp_df[[variable, 'status_group']].groupby(['status_group'])[[variable]].median().reset_index()
    temp_col = temp_df[variable]
    return temp_col


