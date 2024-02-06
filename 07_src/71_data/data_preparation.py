"""
    Data preparation for data analysis
    ----------------------------------
    Encoding data and imputing missing values
    Author : Johary RAMORASATA
"""

# Dependencies
import numpy as np
import pandas as pd
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Import Data
train_data = pd.read_csv("./01_data/11_raw/train.csv", header=0, index_col=0)
test_data = pd.read_csv("./01_data/11_raw/test.csv", header=0, index_col=0)

print(train_data.isna().sum())

# Feature preprocessing
def drop_cols(train_data) :
    out = train_data.copy()
    out.drop(["Ticket"], axis=1, inplace=True)
    out.drop(["Name"], axis=1, inplace=True)
    out.drop(["Cabin"], axis=1, inplace=True)
    out.loc[out["Age"] == 0, "Age"] = np.nan
    out.loc[out["Fare"] == 0, "Fare"] = np.nan
    out.loc[~out["Embarked"].isin(['Q','C','S']), "Embarked"] = np.nan
    return out

train_df = drop_cols(train_data)
test_df = drop_cols(test_data)

train_df, y_train = train_df[train_df.columns.difference(["Survived"])], train_df["Survived"]

# Construction of Imputer
num_columns=["Age","Fare"]
cat_columns=["Sex","Embarked"]

imput1 = SimpleImputer(strategy="most_frequent")
imput2 = IterativeImputer(max_iter=10, random_state=1)
encoder = OneHotEncoder(drop='first', sparse_output=False)

train_df[cat_columns] = pd.DataFrame(imput1.fit_transform(train_df[cat_columns]), index=train_df.index, columns=cat_columns)
test_df[cat_columns] = pd.DataFrame(imput1.transform(test_df[cat_columns]), index=test_df.index, columns=cat_columns)

train_df[num_columns] = pd.DataFrame(imput2.fit_transform(train_df[num_columns].astype(np.float32)), index=train_df.index, columns=num_columns)
test_df[num_columns] = pd.DataFrame(imput2.transform(test_df[num_columns].astype(np.float32)), index=test_df.index, columns=num_columns)

# One Hot Encoding
train_df_encoded = pd.DataFrame(encoder.fit_transform(train_df[cat_columns]), index=train_df.index, columns=encoder.get_feature_names_out(cat_columns))
test_df_encoded = pd.DataFrame(encoder.transform(test_df[cat_columns]), index=test_df.index, columns=encoder.get_feature_names_out(cat_columns))

train_df = pd.concat([train_df, train_df_encoded], axis=1)
test_df = pd.concat([test_df, test_df_encoded], axis=1)

train_df = train_df.drop(cat_columns, axis=1)
test_df = test_df.drop(cat_columns, axis=1)

# Save files to pkl format
with open('./01_data/12_interim/CLEAN_TRAIN_DF.pkl', 'wb') as f:
    pickle.dump(train_df, f)

with open('./01_data/12_interim/Y_TRAIN.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('./01_data/12_interim/CLEAN_TEST_DF.pkl', 'wb') as f:
    pickle.dump(test_df, f)
