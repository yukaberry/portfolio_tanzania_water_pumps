import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from pprint import pprint

from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import pickle
from datetime import datetime
from utils.modeling_utils import timer 

# load data and split train and test
df =pd.read_csv("data/tanzania_cleaned_df2.csv") 
X = df.iloc[0:59400,0:110]
y= df[['status_group']]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
print(df.status_group.value_counts())

# XGB model and its parameters
XGB =XGBClassifier()
pprint(XGB.get_params())

# kfold 5 
kf = KFold(n_splits=5, random_state=42, shuffle=False)

# 
params_xgb = {#'n_estimators': [100], this is default
               'max_depth': [6,8,10],
               #'validate_parameters': [True], this is default
               'min_child_weight': [1,2,3],
               'gamma':[0, 0.5],
               'learning_rate':[0.05,0.1,0.3,0,4],
               'colsample_bytree':[1,0.5]
}

# Scoring ="f1_macro"
grid_no_up = GridSearchCV(XGB, param_grid=params_xgb, cv=kf, 
                          scoring='f1_macro').fit(X_train,y_train)


print(grid_no_up.best_score_)
print(grid_no_up.best_params_)
print(f1_score(y_test, grid_no_up.predict(X_test),average='macro'))
print(grid_no_up.cv_results_)


# scoring = "arrucary"
grid_no_up = GridSearchCV(XGB, param_grid=params_xgb, cv=kf, 
                          scoring='accuracy').fit(X_train_resampled, y_train_resampled)


print(grid_no_up.best_score_)
print(grid_no_up.best_params_)
print(grid_no_up.cv_results_)


# Use OneVsRestClassifier 
xgb_ovr_clf=OneVsRestClassifier(xgboost.XGBClassifier(objective="multi:softmax",num_class=3))
# Get params' key
pprint(xgb_ovr_clf.get_params())
# another way to print params
xgb_ovr_clf.estimator.get_params().keys()
# Set ranges of parameters
# booster types: booster = ['gbtree']
random_grid = {'estimator__n_estimators': [100,200,300], # Number of trees
               'estimator__max_depth': [6,8,10], # Maximum number of levels in tree
               'estimator__validate_parameters': [True], # When set to True, XGBoost will perform validation of input parameters to check whether a parameter is used or not.
               'estimator__min_child_weight': [1,2,3], # the minimum weight (or number of samples if all samples have a weight of 1) required in order to create a new node in the tree.
                                                    #Smaller weight, smaller samples. If too big, will result in overfitiing
               'estimator__gamma':[ 0.0, 0.5], # learning_rate (eta) range: [0,1], default=0.3, prevents overfitting
               'estimator__learning_rate':[0.05,0.1,0.3,0,4],
               'estimator__colsample_bytree':[0.1,0.5]}



# Random search of parameters, using 10 fold cross validation "cv = 5"
# search across 5 different combinations, and use all available cores n_iter = 5
# evaluate accuracy scoring='roc_auc' for multi class
# n_iter :when a big number is set, it will take long to compute! default is 10
# verbose : the high number, show the more details of its process
XGB_random=RandomizedSearchCV(xgb_ovr_clf,param_distributions=random_grid ,
                              n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=10)


# timing starts from this point for "start_time" variable
start_time = timer(None)
XGB_random.fit(X,y)
# timing ends here for "start_time" variable
timer(start_time)
pprint(XGB_random.best_params_)
pprint(XGB_random.best_estimator_)
print(XGB_random.best_score_)
