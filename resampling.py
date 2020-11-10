import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from utils.resampling_utils import resampled_cross_val
from xgboost import XGBClassifier

# read csv
df = pd.read_csv("data/tanzania_cleaned_df2.csv")

# Feature selection 
X = df.iloc[0:59400,0:110]
y= df[['status_group']]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)


# Set parameters 
gridsearched_params_xgb = {'max_depth':10,'colsample_bytree':0.5}
# Set kfold 
kf = KFold(n_splits=5, random_state=42, shuffle=False)

# Reampling X_train y_train + Cross validation (Turned XGB hyperparameters)
XGB =XGBClassifier()
X_train_upsampled,y_train_upsampled,f1_macro_array_scores_xgb_r,f1_macro_score_mean_xgb_r = resampled_cross_val(XGBClassifier, gridsearched_params_xgb , cv=kf)
print(f1_macro_array_scores_xgb_r)
print(f1_macro_score_mean_xgb_r)

# Check resampled train datasets' size
print(y_train_upsampled.status_group.value_counts())
print(X_train_upsampled.shape)

# Save resampled datasets
# Index = Ture for next steps to merge two datasets
X_train_upsampled.to_csv("data/X_train_upsampled.csv",index=True)
X_train_upsampled = pd.read_csv("data/X_train_upsampled.csv")

# Save
y_train_upsampled.to_csv("data/y_train_upsampled.csv",index=True)
y_train_upsampled = pd.read_csv("data/y_train_upsampled.csv")

# Merge X and y train
df_r = pd.merge(X_train_upsampled,y_train_upsampled,how="left",on="Unnamed: 0")


# Drop columns 
to_drop =["Unnamed: 0"]
df_r.drop(to_drop,inplace= True,axis =1)

# save 
df_r.to_csv("data/df_resampled.csv",index=False)
df_r = pd.read_csv("data/df_resampled.csv")
print(df_r.head())
print(df_r.shape)

