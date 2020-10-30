import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN


def split_data_resampling(X,y,test_percentage=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=42)
    smote_enn = SMOTEENN(random_state=0)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled,  X_test, y_test 

