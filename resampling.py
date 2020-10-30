import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from utils.resampling_utils import split_data_resampling

# read csv
df = pd.read_csv("data/tanzania_cleaned_df.csv")

# Feature selection 
X = df.iloc[0:59400,0:110]
y= df.iloc[0:59400,-1]

# split data and resampled only train datasets
X_train_resampled, y_train_resampled, X_test, y_test  = split_data_resampling(X,y)
# see returned values
print(df.label.value_counts())
print(y_train_resampled.value_counts())
print(y_test.value_counts())



