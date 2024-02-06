"""
    Feature engineering for data analysis
    -------------------------------------
    Feature selection and processing to optimize training for ML Models
"""

# Dependencies
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_selection import SequentialFeatureSelector, f_classif, f_regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Import data
with open('./01_data/12_interim/CLEAN_TRAIN_DF.pkl', 'rb') as file:
    train_df = pd.read_pickle(file)

with open('./01_data/12_interim/Y_TRAIN.pkl', 'rb') as file:
    y_train = pd.read_pickle(file)


# Ranking features
cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)
estimator = RandomForestClassifier()
feature_dict = {}

n = len(train_df.columns)

for n_features in range(3,n) :
    selector = SequentialFeatureSelector(estimator=estimator, n_features_to_select=n_features,
                                        direction="backward", scoring="accuracy", cv=cv,
                                        n_jobs=-1)

    selector.fit(train_df, y_train)

    name = type(estimator).__name__
    feature_list = selector.get_feature_names_out()
    feature_dict[f"{name}_{n_features}"] = feature_list

    print(f"Features number : {n_features} for model {type(estimator).__name__}")
    print(f"Features : {selector.get_feature_names_out()}")
    
    train_df_optim = train_df[feature_list]

    estimator.fit(train_df_optim, y_train)
    y_predict = estimator.predict(train_df_optim)
    print(f"Accuracy : {accuracy_score(y_predict, y_train):.4f} -- ",
          f"Precision : {precision_score(y_predict, y_train):.4f} -- ",
          f"Recall : {recall_score(y_predict, y_train):.4f}")


# Save feature dictionnary
with open(f"./01_data/12_interim/{name}_feature_dict.pkl", "wb") as file :
    pickle.dump(feature_dict, file)