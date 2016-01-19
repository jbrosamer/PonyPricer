import re, glob

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# Path of data
path="/Users/jbrosamer/PonyPricerFiles/Batchs/DressageId*1000*.csv"     
    
def all_data(path):
    """
    Takes in: with wildcarding of dataframes stored in .csv
    
    Returns: a dataframe of all 
    """
    files=glob.glob(path)
    dfs=[]
    frame=pd.DataFrame()
    for f in files:
        dfs.append(pd.read_csv(f))
    frame = pd.concat(dfs)
    frame = frame.reset_index().drop('index', axis = 1)
    return frame
def predictPrice(age=10, breed='Westphalian', gender='Gelding', height='17.0hh'):
    return 5,000

def clean_col(df):
    """
    Takes in: dataframe from all_data

    Returns: a dataframe after converting null values
    """
    for col in df.columns:
        # Honda civics are 4 cylinders
        if col == 'cylinders':
            df.loc[pd.isnull(df[col]), col] = 4
        elif col == 'extra':
            df.loc[pd.isnull(df[col]), col] = 'normal'
        elif col == 'fuel':
            df.loc[pd.isnull(df[col]), col] = 'gas'
        elif col == 'drive':
            df.loc[pd.isnull(df[col]), col] = 'fwd'
        elif col == 'transmission':
            df.loc[pd.isnull(df[col]), col] = 'auto'
        elif col == 'type':
            df.loc[pd.isnull(df[col]), col] = 'unknown'
    return df
        
def encode(df):
    """
    Takes in: dataframe from clean_col
    
    Returns: a dataframe that LabelEncodes the categorical variables
    """
    columns = ['drive', 'extra', 'fuel',
               'odometer', 'title_stat',
               'transmission', 'type']
    for col in columns:
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
    
    final_cols = ['year', 'cylinders', 'drive', 'extra',
                  'fuel', 'odometer', 'title_stat',
                  'transmission', 'type', 'price']
    # Order columns with price as the last column
    df = df[final_cols]
    return df

class Model():
    
    def __init__(self, df, params, test_size = 0.3):
        self.df = df
        self.params = params
        self.test_size = 0.3
        
    def split(self):
        np.random.seed(1)
        self.df = self.df.reindex(np.random.permutation(self.df.index))
        self.df = self.df.reset_index().drop('index', axis = 1)
        X = self.df.as_matrix(self.df.columns[:-1])
        y = self.df.as_matrix(['price'])[:,0]
        X_train, X_test, y_train, y_test = train_test_split(
                                                X, y,
                                                test_size=self.test_size,
                                                )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def kfold_cv(self, n_folds = 3):
        """
        Takes in: number of folds
        
        Prints out RMSE score and stores the results in self.results
        """

        cv = KFold(n = self.X_train.shape[0], n_folds = n_folds)
        gbr = GradientBoostingRegressor(**self.params)
        self.rmse_cv = []
        self.results = {'pred': [],
                   'real': []}
        
        for train, test in cv:
            gbr.fit(self.X_train[train], self.y_train[train])
            pred = gbr.predict(self.X_train[test])
            error = mean_squared_error(pred, self.y_train[test])**0.5
            self.results['pred'] += list(pred)
            self.results['real'] += list(self.y_train[test])
            self.rmse_cv += [error]
        print 'RMSE Scores:', self.rmse_cv
        print 'Mean RMSE:', np.mean(self.rmse_cv)
        
    def plot_results(self):
        """
        Plots results from CV
        """
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (12,10))

        ax.scatter(self.results['real'], self.results['pred'], color = (0.6,0.0,0.2),
                   label = 'Model Predictions',
                   s = 100, alpha = 0.4)
        ax.plot(np.arange(0, 50000),np.arange(0, 50000), color = 'black',
                   label = 'Perfect Prediction Line',
                   lw = 4, alpha = 0.5, ls = 'dashed')

        ax.set_xlabel('Actual Price ($)',fontsize = 20)
        ax.set_ylabel('Predicted Price ($)', fontsize = 20)
        ax.set_title('Results from KFold Cross-Validation', fontsize = 25)
        ax.set_xlim(0,30000)
        ax.set_ylim(0,30000)
        ax.legend(loc=2, fontsize = 16)
        ax.tick_params(labelsize =20)
    
    def validate(self):
        """
        Validate Model on Test set
        """
        gbr = GradientBoostingRegressor(**self.params)
        gbr.fit(self.X_train, self.y_train)
        self.preds = gbr.predict(self.X_test)
        self.rmse = mean_squared_error(self.preds, self.y_test)**0.5
        print 'RMSE score:', self.rmse
        
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (12,10))

        ax.scatter(self.y_test, self.preds, color = (0.6,0.0,0.2),
                   label = 'Model Predictions',
                   s = 100, alpha = 0.4)
        ax.plot(np.arange(0, 50000),np.arange(0, 50000), color = 'black',
                   label = 'Perfect Prediction Line',
                   lw = 4, alpha = 0.5, ls = 'dashed')

        ax.set_xlabel('Actual Price ($)',fontsize = 20)
        ax.set_ylabel('Predicted Price ($)', fontsize = 20)
        ax.set_title('Results from Test Set', fontsize = 25)
        ax.set_xlim(0,30000)
        ax.set_ylim(0,30000)
        ax.legend(loc=2, fontsize = 16)
        ax.tick_params(labelsize =20)
    def run():
        """
            Run the test
        """
        df_test = all_data(path)
        df = df_test.copy()
        df = clean_col(df)
        df = encode(df)
        params_gbr = {'loss': 'ls',
                      'learning_rate': 0.02,
                      'n_estimators': 500,
                      'max_depth': 6,
                      'min_samples_split': 2,
                      'min_samples_leaf': 13,
                      'subsample': 0.7
                     }
        b = Model(df, params = params_gbr)
        b.split()
        b.kfold_cv(n_folds = 3)