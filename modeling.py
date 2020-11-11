import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import scikitplot as skplt
from pprint import pprint

from utils.modeling_utils import modeling
from utils.modeling_utils import return_x_val_accuracy
from utils.modeling_utils import return_x_val_f1_macro

import pickle


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

# load data and split train and test (Resampled dataset)
# Split X_train_resampled and y_train_resampled
# Only split X and y.(no train and test) This dataset is only for training due to upsampled data. 
# Test dataset needs to be unseen
df_r =pd.read_csv("data/df_resampled.csv")
X_train_resampled = df_r.iloc[0:61959,0:110]
y_train_resampled = df_r[["status_group"]]
print(len(X_train_resampled))
print(len(y_train_resampled))
print(y_train_resampled.status_group.value_counts())

Basemodels used no upsampled data, holdout method (8:2) validation
XGB =XGBClassifier()
XGB_clf_report, XGB_clf_acc_score, XGB_f1score,XGB_y_pred = modeling(XGB,X_train,y_train,X_test,y_test)

GBC = GradientBoostingClassifier()
GBC_clf_report, GBC_clf_acc_score, GBC_f1score,GBC_y_pred = modeling(GBC,X_train,y_train,X_test,y_test)


RFC =RandomForestClassifier()
RFC_clf_report, RFC_clf_acc_score, RFC_f1score,RFC_y_pred = modeling(RFC,X_train,y_train,X_test,y_test)

svc =SVC()
svc_clf_report, svc_clf_acc_score, svc_f1score,svc_y_pred = modeling(svc,X_train,y_train,X_test,y_test)

results={'Classifier':["XGB Classifier","Gradient Boosting Classifier","Random Forest Classifier","SVC"],
'Accuracy':[str(XGB_clf_acc_score)[:5],str(GBC_clf_acc_score)[:5],str(RFC_clf_acc_score)[:5],str(svc_clf_acc_score)[:5]],
'F1_macro':[str(XGB_f1score)[:5],str(GBC_f1score)[:5],str(RFC_f1score)[:5],str(svc_f1score)[:5]]}

score_report_df =pd.DataFrame(data=results,columns=["Classifier","Accuracy","F1_macro"])
print("Base models holdout method (8:2) validation")
print(score_report_df)



# 5 folds cross val, f1 macro and accuracy, baseline without resampling
cross_validation_score_base_xgb,estimated_accuracy_base_xgb = return_x_val_accuracy(XGB,X_train,y_train)
cross_validation_score_base_gbc,estimated_accuracy_base_gbc = return_x_val_accuracy(GBC,X_train,y_train)
cross_validation_score_base_rfc,estimated_accuracy_base_rfc = return_x_val_accuracy(RFC,X_train,y_train)
cross_validataion_score_base_svc,estimated_accuracy_base_svc = return_x_val_accuracy(svc,X_train,y_train)

cross_validation_score_f1_base_xgb,estimated_f1_base_xgb = return_x_val_f1_macro(XGB,X_train,y_train)
cross_validation_score_f1_base_gbc,estimated_f1_base_gbc = return_x_val_f1_macro(GBC,X_train,y_train)
cross_validation_score_f1_base_rfc,estimated_f1_base_rfc = return_x_val_f1_macro(RFC,X_train,y_train)
cross_validation_score_f1_base_svc,estimated_f1_base_svc = return_x_val_f1_macro(svc,X_train,y_train)

results_x_val={'Classifier':["XGB Classifier","Gradient Boosting Classifier","Random Forest Classifier","SVC"],
'Cross validation accuracy':[str(estimated_accuracy_base_xgb)[:5],str(estimated_accuracy_base_gbc)[:5],str(estimated_accuracy_base_rfc)[:5],str(estimated_accuracy_base_svc)[:5]],
'Cross validation F1_macro':[str(estimated_f1_base_xgb)[:5],str(estimated_f1_base_gbc)[:5],str(estimated_f1_base_rfc)[:5],str(estimated_f1_base_svc)[:5]]}

score_report_df_x_val_base =pd.DataFrame(data=results_x_val,columns=['Classifier','Cross validation accuracy','Cross validation F1_macro'])

print("Base models 5 folds cross validation (Train_Test Split 8:2)")
print(score_report_df_x_val_base)

# Basemodels used upsampled train dataset, holdout method (8:2) validation
XGB_clf_report, XGB_clf_acc_score, XGB_f1score,XGB_y_pred = modeling(XGB,X_train_resampled,y_train_resampled,X_test,y_test)
GBC_clf_report, GBC_clf_acc_score, GBC_f1score,GBC_y_pred = modeling(GBC,X_train_resampled,y_train_resampled,X_test,y_test)
RFC_clf_report, RFC_clf_acc_score, RFC_f1score, RFC_y_pred = modeling(RFC,X_train_resampled,y_train_resampled,X_test,y_test)
svc_clf_report, svc_clf_acc_score, svc_f1score,svc_y_pred = modeling(svc,X_train_resampled,y_train_resampled,X_test,y_test)

results={'Classifier':["XGB Classifier","Gradient Boosting Classifier","Random Forest Classifier","SVC"],
'Accuracy':[str(XGB_clf_acc_score)[:5],str(GBC_clf_acc_score)[:5],str(RFC_clf_acc_score)[:5],str(svc_clf_acc_score)[:5]],
'F1_macro':[str(XGB_f1score)[:5],str(GBC_f1score)[:5],str(RFC_f1score)[:5],str(svc_f1score)[:5]]}

score_report_df =pd.DataFrame(data=results,columns=["Classifier","Accuracy","F1_macro"])
print("Base models with upsampled train datasets,  holdout method (8:2) validation")
print(score_report_df)


# Turned hyperparameter and WTIH resampled dataset

# Print baseline hyperparameters
pprint(XGB.get_params())
# Pass turned hyperparameters
# when you set objective="multi:softmax", num_class=3 (func, non func, need repair) needs to be set manually! 
# get_params().keys() does NOT SHOW the param! 
XGB_t= XGBClassifier(colsample_bylevel=0.5,max_depth=10,objective="multi:softmax",num_class=3)
pprint(XGB_t.get_params())

XGB_t_clf_report_r, XGB_t_clf_acc_score_r, XGB_t_f1score_r, XGB_t_y_pred_r= modeling(XGB_t,X_train_resampled,y_train_resampled,X_test,y_test)
print(XGB_t_clf_report_r)

# Confusion matrix visualisation
XGB_t_confusion_matrix_ =confusion_matrix(y_test,XGB_t_y_pred_r)
class_names = ["Func","Need Repair","Non Func"]
fig,ax =plot_confusion_matrix(conf_mat = XGB_t_confusion_matrix_,colorbar = True,
                             show_absolute=False, show_normed=True,
                             class_names = class_names)
plt.show()

# ROC and AUC curve chart visualisation
predicted_probas = XGB_t.fit(X_train_resampled,y_train_resampled).predict_proba(X_test)
# print(predicted_probas.shape)
# print(predicted_probas)
y_test_arr = np.ravel(y_test)
# print(y_test_arr.shape)
# print(y_test_arr)
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

# Turned hyperparameter and no resampled dataset
XGB_t_clf_report_r, XGB_t_clf_acc_score_r, XGB_t_f1score_r,XGB_t_y_pred_r = modeling(XGB_t,X_train,y_train,X_test,y_test)
#print(XGB_t_clf_report_r)

# Confusion matrix
XGB_t2_confusion_matrix_ =confusion_matrix(y_test,XGB_t_y_pred_r)
class_names = ["Func","Need Repair","Non Func"]
fig,ax =plot_confusion_matrix(conf_mat = XGB_t2_confusion_matrix_,colorbar = True,
                             show_absolute=False, show_normed=True,
                             class_names = class_names)
plt.show()

# ROC and AUC curve chart
predicted_probas2 = XGB_t.fit(X_train,y_train).predict_proba(X_test)
# print(predicted_probas2.shape)
# print(predicted_probas2)
skplt.metrics.plot_roc(y_test, predicted_probas2)
plt.show()



# Save a model
# set best parameters from grid search
XGB_t= XGBClassifier(colsample_bylevel=0.5,max_depth=10,objective="multi:softmax",num_class=3)
# save the model to disk
pickle_model = 'model/turned_xgb.sav'
pickle.dump(XGB_t, open(pickle_model, 'wb'))
# load "picked_model" and define as "loaded_model" 
loaded_model = pickle.load(open(pickle_model,'rb'))

