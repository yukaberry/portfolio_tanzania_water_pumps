import pandas as pd
import numpy as np
from resampling import X_train_resampled, y_train_resampled
from resampling import X_test, y_test 

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def modeling(clf,X_train_resampled,y_train_resampled,X_test):
    clf.fit(X_train_resampled,y_train_resampled)
    y_pred =clf.predict(X_test)
    clf_report = classification_report(y_test,y_pred)
    clf_acc_score = accuracy_score(y_test, y_pred)
    f1score=f1_score(y_test, y_pred, average='macro')
    return  clf_report, clf_acc_score, f1score