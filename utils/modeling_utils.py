import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from datetime import datetime


def modeling(clf,X_train,y_train,X_test,y_test):
    """train a model and return model's accuracy and F1_macro"""

    clf.fit(X_train,y_train)
    y_pred =clf.predict(X_test)
    clf_report = classification_report(y_test,y_pred)
    clf_acc_score = accuracy_score(y_test, y_pred)
    f1score=f1_score(y_test, y_pred, average='macro')
    return  clf_report, clf_acc_score, f1score, y_pred


def return_x_val_accuracy(clf,X_train,y_train,kfold=5):
    """Return kfold cross validation scores of accuracy and its mean. kfold = 5 as default """
    cross_validation_score = cross_val_score(clf,X_train,y_train,cv=kfold)
    estimated_accuracy = (cross_validation_score.mean(), cross_validation_score.std() * 2)
    
    print("Estimate cross validation accuracy: %0.2f (+/- %0.2f)" % (cross_validation_score.mean(), cross_validation_score.std() * 2))
    print("Cross validation scores " + str(cross_validation_score))
    return cross_validation_score,estimated_accuracy


def return_x_val_f1_macro(clf,X_train,y_train,kfold=5):
    """Return kfold cross validation scores of f1_macro and its mean kfold = 5 as default """
    cross_validation_score_f1 = cross_val_score(clf,X_train,y_train,cv=kfold,scoring='f1_macro')
    estimated_f1 = (cross_validation_score_f1.mean(), cross_validation_score_f1.std() * 2)
    
    print("Estimate cross validation F1_macro: %0.2f (+/- %0.2f)" % (cross_validation_score_f1.mean(), cross_validation_score_f1.std() * 2))
    print("F1_macro Cross validation scores: " + str(cross_validation_score_f1))
    return cross_validation_score_f1,estimated_f1
    


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))