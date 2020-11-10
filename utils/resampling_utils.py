import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

from sklearn.metrics import f1_score

# read csv
df = pd.read_csv("data/tanzania_cleaned_df2.csv")

# Feature selection 
X = df.iloc[0:59400,0:110]
y= df[['status_group']]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)


def split_data_resampling(X,y,test_percentage=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=42)
    smote_enn = SMOTEENN(random_state=0)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled,  X_test, y_test 


def resampled_cross_val(model, params, cv=None):
    """
    Creates folds manually, and upsamples within each fold.
    Returns an array of validation f1_macro scores
    """
    if cv is None:
        cv = KFold(n_splits=5, random_state=42)

    smoter = SMOTE(random_state=42)
    
    f1_macro_scores = []
    for train_fold_index, val_fold_index in KFold(n_splits=5, random_state=42).split(X_train, y_train):
        # Get the training data
        X_train_fold, y_train_fold = X_train.iloc[train_fold_index], y_train.iloc[train_fold_index]
        # Get the validation data
        X_val_fold, y_val_fold = X_train.iloc[val_fold_index], y_train.iloc[val_fold_index]
        
         # Upsample only the data in the training section
        X_train_upsampled, y_train_upsampled = smoter.fit_resample(X_train_fold,
                                                                           y_train_fold)
         # Fit the model on the upsampled training data
        model_upsample = model(**params).fit(X_train_upsampled,  y_train_upsampled)
        # Score the model on the (non-upsampled) validation data
        f1_macro_score = f1_score(y_val_fold, model_upsample.predict(X_val_fold),average='macro')
        f1_macro_scores.append(f1_macro_score)
        f1_macro_score_mean=f1_macro_score.mean()
        f1_macro_array_scores = np.array(f1_macro_scores)
        

    return X_train_upsampled,y_train_upsampled,f1_macro_array_scores,f1_macro_score_mean

