"""
    Model selection and training
    ----------------------------
    The objective here is to train and fine-tune models for classification purpose
"""

# Dependencies
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, log_loss,make_scorer

# Define the estimator to train
estimator = RandomForestClassifier()

# Import data
with open('./01_data/12_interim/CLEAN_TRAIN_DF.pkl', 'rb') as file:
    train_df = pd.read_pickle(file)

with open('./01_data/12_interim/Y_TRAIN.pkl', 'rb') as file:
    y_train = pd.read_pickle(file)

with open('./01_data/12_interim/CLEAN_TEST_DF.pkl', 'rb') as file:
    test_df = pd.read_pickle(file)
test_df = test_df[train_df.columns]

# Import optimized feature dict
with open(f"./01_data/12_interim/{type(estimator).__name__}_feature_dict.pkl", 'rb') as file:
    feature_dict = pd.read_pickle(file)

# Utils for GridSearch algorithm
cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)

# Param_grid for RANDOM FOREST
param_grid = {
    'n_estimators': [30, 50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True],
    'criterion': ['gini']
}

# Metrics
acc = make_scorer(accuracy_score)
prec = make_scorer(precision_score)
rec = make_scorer(recall_score)
f1 = make_scorer(f1_score)
fbeta = make_scorer(fbeta_score)
logl = make_scorer(log_loss)

metricsList = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'log_loss': log_loss,
}

"""

# GridSearch algorithm WITHOUT feature engineering
feat = False

grid = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=acc, cv=cv, n_jobs=-1)
grid.fit(X=train_df, y=y_train)

print(f"Best accuracy for model {type(estimator).__name__} : {grid.best_score_}")
print("\n") ; print(f"Best Parameters : {grid.best_params_}") ; print("\n")

"""

# GridSearch algorithm WITH feature engineering
feat = True

with open(f"./01_data/12_interim/{type(estimator).__name__}_feature_dict.pkl", "rb") as file :
    feature_dict = pickle.load(file)

train_df_optim = train_df[feature_dict[f"{type(estimator).__name__}_6"]]
test_df_optim = test_df[feature_dict[f"{type(estimator).__name__}_6"]]

grid = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=acc, cv=cv, n_jobs=-1)
grid.fit(X=train_df_optim, y=y_train)

print(f"Best accuracy for model {type(estimator).__name__} : {grid.best_score_}")
print("\n") ; print(f"Best Parameters : {grid.best_params_}") ; print("\n")


## Saving model
best_model = grid.best_estimator_

with open(f"./03_models/{type(estimator).__name__}_best_model{"_feat_eng" if feat else ''}.pkl", "wb") as file :
    pickle.dump(best_model, file)


# Metrics evaluation
y_valid = best_model.predict(train_df_optim if feat else train_df)
y_test = pd.DataFrame({"PassengerId" : test_df.index.astype(np.int32),
                       "Survived" : best_model.predict(test_df_optim if feat else test_df)
                       })

with open(f"./01_data/13_processed/{type(estimator).__name__}_{"feat_eng_" if feat else ''}Prediction.csv",'w') as file :
    y_test.to_csv(file, index=False, header=True)

for metricName, metricFunction in metricsList.items():
    print(f"{metricName} : {metricFunction(y_valid, y_train):.4f}")

