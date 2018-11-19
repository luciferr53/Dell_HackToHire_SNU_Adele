
# coding: utf-8

# In[377]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import xgboost as xg


# In[378]:


df = pd.read_csv('dataset.csv')


# In[379]:


del df['PRODUCT ID']
del df['Customer Id']
del df['customer name']


# In[380]:


df.head()


# In[381]:


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[387]:


features_df = ['CPU']
df = MultiColumnLabelEncoder(columns = ['CPU']).fit_transform(df)
#df


# In[383]:


X = df[features_df]
y = df['quantiy sold'].as_matrix()


# In[384]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[385]:


#regr = linear_model.LinearRegression()# Train the model using the training sets
regr = Ridge(alpha=1.0)
regr.fit(X_train, y_train)
#regr.coef_
joblib.dump(regr, 'trained_house_classifier_model.pkl')


# In[386]:


#y_pred = regr.predict(X_test)
#y_pred


# In[375]:


#y_test


# In[376]:


'''from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))
rms
#rms = sqrt(mean_squared_error(X_true))
'''
