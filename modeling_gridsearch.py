import pandas as pd
import numpy as np
from resampling import X_train_resampled, y_train_resampled
# from resampling import X_test, y_test 
# from resampling import  X,y
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pprint import pprint

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import pickle

XGB =XGBClassifier()
pprint(XGB.get_params())


"""""""Randomised Search""""""" 
# Use the random grid to search for best hyperparameters


# Number of trees
n_estimator = [int(x) for x in np.linspace(start = 100, stop =500, num = 5)]

# booster types
booster = ['gbtree']

# Maximum number of levels in tree
max_dep = [int(x) for x in np.linspace(4, 16, num = 6)]
max_dep.append(None)

# When set to True, XGBoost will perform validation of input parameters to check whether a parameter is used or not.
validate_parameter = [True, False]

# the minimum weight (or number of samples if all samples have a weight of 1) required in order to create a new node in the tree.
#Smaller weight, smaller samples. If too big, will result in overfitiing
min_child_wei = [int(x) for x in np.linspace(1, 5, num = 5)]
min_child_wei.append(None)

# learning_rate (eta) range: [0,1], default=0.3, prevents overfitting
eta = [0.2,0.3,0.4]

# Method of selecting samples for training each tree
bootstp = [True]

random_grid = {'n_estimators': n_estimator,
               'booster':booster,
               'max_depth': max_dep,
               'validate_parameters': validate_parameter,
               'min_child_weight': min_child_wei,
               'learning_rate':eta,
               'bootstrap': bootstp}


random_grid = {'n_estimators': [100,200,300],
               'booster':'gbtree',
               'max_depth': [8,10,12,14],
               'validate_parameters': [True, False],
               'min_child_weight': [1],
               'learning_rate':[0.2,0.3,0.4],
               'bootstrap': [True]}



# Random search of parameters, using 10 fold cross validation "cv = 5"
# search across 100 different combinations, and use all available cores n_iter = 100
# evaluate accuracy scoring='f1_macro' for multi class
# n_iter :when a big number is set, it will take long to compute! default is 10
# verbose : the high number the more details of its outcome
print("start randomiseserch")
XGB_random = RandomizedSearchCV(estimator=XGB, param_distributions=random_grid,
                              n_iter = 10, scoring='f1_macro', 
                              cv = 3, verbose=10, random_state=42, n_jobs=-1,
                              return_train_score=True)
print("fitting model")
# Fit the random search model
XGB_random.fit(X_train_resampled,y_train_resampled)

print(XGB_random.best_params_)
print(XGB_random.best_score_)




"""
""""""Grid Search""""""
# Create the parameter grid based on the results of random search 

grids_params = {}



xgb_gridsearch = GridSearchCV(estimator = XGB, param_grid = grids_params, scoring='f1_macro',
                          cv = 5, n_jobs = -1, verbose = 2, return_train_score=True)


xgb_gridsearch.fit(X_train_resampled,y_train_resampled)
print(xgb_gridsearch.best_params_)
print(xgb_gridsearch.best_socre_)

""""""Save a model""""""
# set best parameters from grid search
turned_xgb = XGBClassifier()
# save the model to disk
pickle_model = 'model/turned_xgb.sav'
pickle.dump(turned_xgb, open(pickle_model, 'wb'))

""""""Cross Validation"""""""
# load the model 
loaded_model = pickle.load(open(pickle_model,'rb'))

# Use X and y instead of splited train and test (X_train_resampled,y_train_resampled)
cross_validation_score_grid = cross_val_score(loaded_model, X, y)
print("Estimate the Accuracy of XGB classifier(cross validation score): %0.2f (+/- %0.2f)" % (cross_validation_score_grid.mean(), cross_validation_score_grid.std() * 2))

cross_val_score(loaded_model,X, y,cv =5,scoring='f1_macro')
cross_validation_score_f1_grid = cross_val_score(loaded_model, X, y,scoring='f1_macro')
print("Estimate F1-macro of XGB classifier(cross validation score): %0.2f (+/- %0.2f)" % (cross_validation_score_f1_grid.mean(), cross_validation_score_f1_grid.std() * 2))


y_pred =loaded_model.predict(X_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
"""
"""
# Random search parameters
xgb_ran = XGBClassifier(min_child_weight =1,max_depth=10,bootstrap=True,booster="gbtree",
                       n_estimators=200,
                       validate_parameters=True,
                       learning_rate=0.3)
pprint(xgb_ran.get_params())

cross_validation_score = cross_val_score(xgb_ran, X, y)
print(cross_validation_score)
print("Estimate the Accuracy: %0.2f (+/- %0.2f)" % (cross_validation_score.mean(), cross_validation_score.std() * 2))

cross_validation_score_f1 = cross_val_score(xgb_ran, X, y,scoring='f1_macro')
print(cross_validation_score_f1)
print("Estimate F1-macro: %0.2f (+/- %0.2f)" % (cross_validation_score_f1.mean(), cross_validation_score_f1.std() * 2))
"""



# print confusion matrix 
# save image and insert it on readme 
