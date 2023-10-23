import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'category_encoders'])  #replaces !pip install
import category_encoders as ce

class CustomMappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    placeholder = "NaN"
    column_values = X[self.mapping_column].fillna(placeholder).tolist()  #convert all nan values to the string "NaN" in new list
    column_values = [np.nan if v == placeholder else v for v in column_values]  #now convert back to np.nan
    keys_values = self.mapping_dict.keys()

    column_set = set(column_values)  #without the conversion above, the set will fail to have np.nan values where they should be.
    keys_set = set(keys_values)      #this will have np.nan values where they should be so no conversion necessary.

    #now check to see if all keys are contained in column.
    keys_not_found = keys_set - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

    #do actual mapping
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
  #your __init__ method below
  def __init__(self, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict

  #define fit to do nothing but give warning
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return self

  #write the transform method with asserts. Again, maybe copy and paste from MappingTransformer and fix up.
  def transform(self, X):
    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_dict], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__} constructor expected DataFrame but got {type(X)} instead.'

    columns_values = X.columns
    keys_values = self.mapping_dict.keys()

    column_set = set(columns_values)
    keys_set = set(keys_values)

    keys_not_found = keys_set - column_set
    assert not keys_not_found, f"{self.__class__.__name__}[{self.mapping_dict}] these mapping keys do not appear in the column: {keys_not_found}\n"

    #now check to see if some keys are absent
    keys_absent = column_set - keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_dict}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

      #do actual mapping
    X_ = X.copy()
    X_.rename(columns=self.mapping_dict, inplace=True)
    return X_
  #write fit_transform that skips fit
  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column: str, dummy_na=False, drop_first=False):
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  #fill in the rest below
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return self

  def transform(self, X):
      #Set up for producing warnings. First have to rework nan values to allow set operations to work.
      #In particular, the conversion of a column to a Series, e.g., X[self.mapping_dict], transforms nan values in strange ways that screw up set differencing.
      #Strategy is to convert empty values to a string then the string back to np.nan
      assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__} constructor expected DataFrame but got {type(X)} instead.'
      assert self.target_column in X.columns.to_list(), f'Column "{self.target_column}" not exist.'

        #do actual mapping
      X_ = X.copy()
      X_ = pd.get_dummies(X_, prefix=self.target_column, prefix_sep='_', columns=[self.target_column], dummy_na=self.dummy_na, drop_first=self.drop_first)
      return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):
    self.target_column = target_column
    self.lowb = None
    self.upb = None
  def fit(self, df):
    assert isinstance(df, pd.core.frame.DataFrame), f'expected Dataframe but got {type(df)} instead.'
    assert self.target_column in df.columns, f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in df[self.target_column].to_list()])

    mean = (df[self.target_column]).mean()
    sigma = (df[self.target_column]).std()

    self.lowb = mean - (3 * sigma)
    self.upb = mean + (3 * sigma)
    return self
  def transform(self, df):
    assert self.lowb and self.upb, f'This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    df_ = df.copy()
    df_[self.target_column] = df_[self.target_column].clip(lower=self.lowb, upper=self.upb)
    df_.reset_index()
    return df_
  def fit_transform(self, df):
    self.fit(df)
    result = self.transform(df)
    return result
    
class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    self.target_column = target_column
    self.fence = 1.5 if fence == 'inner' else 3
    self.lowb = None
    self.upb = None
  def fit(self, df):
    q1 = df[self.target_column].quantile(0.25)
    q3 = df[self.target_column].quantile(0.75)
    iqr = q3-q1
    self.lowb = q1-self.fence*iqr
    self.upb = q3+self.fence*iqr

    return
  def transform(self, df):
    assert self.lowb and self.upb, f'This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    df_ = df.copy()
    df_[self.target_column] = df_[self.target_column].clip(lower=self.lowb, upper=self.upb)
    df_.reset_index()
    return df_
  def fit_transform(self, df, y=None):
    self.fit(df)
    result = self.transform(df)
    return result

class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column):
    self.column = column
    self.iqr = None
    self.median = None
  def fit(self, df):
    self.iqr = df[self.column].quantile(.75) - df[self.column].quantile(.25)
    self.median = df[self.column].median()
  def transform(self, df):
    assert self.iqr != None and self.median != None, f'This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    df_ = df.copy()
    df_[self.column] = (df_[self.column] - self.median) / self.iqr
    return df_
  def fit_transform(self, df, y = None):
    self.fit(df)
    print(self.iqr, self.median)
    result = self.transform(df)
    return result

def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier(n_neighbors=5)
  var = []
  for i in range(1, n):
    train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                    random_state=i, stratify=labels)
    model.fit(train_X, train_y)  #train model
    train_pred = model.predict(train_X)           #predict against training set
    test_pred = model.predict(test_X)             #predict against test set
    train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
    test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
    f1_ratio = test_f1/train_f1          #take the ratio
    var.append(f1_ratio)

  rs_value = sum(var)/len(var)  #get average ratio value
  idx = np.array(abs(var - rs_value)).argmin()
  return idx
