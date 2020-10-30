import pandas as pd
import numpy as np
from resampling import X_train_resampled, y_train_resampled
from resampling import X_test, y_test 

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from utils.modeling_utils import modeling

# Modeling using "modeling" function 
# classification_report, accuracy_score and  f1_score from each models 
XGB =XGBClassifier()
XGB_clf_report, XGB_clf_acc_score, XGB_f1score = modeling(XGB,X_train_resampled,y_train_resampled,X_test)

GBC = GradientBoostingClassifier()
GBC_clf_report, GBC_clf_acc_score, GBC_f1score = modeling(GBC,X_train_resampled,y_train_resampled,X_test)


RFC =RandomForestClassifier()
RFC_clf_report, RFC_clf_acc_score, RFC_f1score = modeling(RFC,X_train_resampled,y_train_resampled,X_test)

svc =SVC()
svc_clf_report, svc_clf_acc_score, svc_f1score = modeling(svc,X_train_resampled,y_train_resampled,X_test)

# Save the results in DataFrame
results={'Classifier':["XGB Classifier","Gradient Boosting Classifier","Random Forest Classifier","SVC"],
'Accuracy':[str(XGB_clf_acc_score)[:5],str(GBC_clf_acc_score)[:5],str(RFC_clf_acc_score)[:5],str(svc_clf_acc_score)[:5]],
'F1_macro':[str(XGB_f1score)[:5],str(GBC_f1score)[:5],str(RFC_f1score)[:5],str(svc_f1score)[:5]]}

score_report_df =pd.DataFrame(data=results,columns=["Classifier","Accuracy","F1_macro"])
print(score_report_df)
