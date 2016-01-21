import re, glob, pickle

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import median_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import  stats

final_cols = ['age', 'gender', 'inches', 'color', 'breed']# 'color', 'registered', 'price']
final_cols+=['lnprice']
lblColumns=['breed', 'color', 'gender']
# Path of data
priceMin=1000
priceMax=100000
pandasPath="/Users/jbrosamer/PonyPricer/BatchBkup/DressageAllAds.p"
    
def all_data(path=pandasPath):
    """
    Takes in: with wildcarding of dataframes stored in .csv
    
    Returns: a dataframe of all 
    """
    df=pickle.load(open(pandasPath, 'rb'))    
    return df
def clean_col(df):
    print "df.columns",df.columns
    df=df[(df['age']>0) & (df['price']>=priceMin) &  (df['price']<=priceMax) & (df['inches']>50) & (df['gender'] != '')]
    df = df.reset_index().drop('index', axis = 1)
    return df




def encode(df):
    """
    Takes in: dataframe from clean_col
    
    Returns: a dataframe that LabelEncodes the categorical variables
    """

    for col in lblColumns:
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
    # Order columns with price as the last column
    df = df[final_cols]
    return df




class Model():
    
    def __init__(self, df=None, params={}, test_size = 0.3):
        if df is None:
            df=all_data()
        self.df = df
        self.params = params
        self.test_size = 0.3
        
    def split(self):
        np.random.seed(1)
        self.df = self.df.reindex(np.random.permutation(self.df.index))
        self.df = self.df.reset_index().drop('index', axis = 1)
        X = self.df.as_matrix(self.df.columns[:-1])
        y = self.df.as_matrix(['lnprice'])[:,0]
        X_train, X_test, y_train, y_test = train_test_split(
                                                X, y,
                                                test_size=self.test_size,
                                                )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.gbr=None

    def makeModel(self):
        gbr = GradientBoostingRegressor(**self.params)
        self.X=self.df.as_matrix(self.df.columns[:-1])
        self.Y=self.df.as_matrix(['lnprice'])[:,0]

        gbr.fit(self.X, self.Y)
        self.gbr=gbr
        return gbr
    # def predictPrice(self, breed="Thoroughbred", age=10., height=66., gender="Gelding"):
    #     inDf=pd.DataFrame(columns=final_cols)
    #     inDf['']

        
    def kfold_cv(self, n_folds = 3):
        """
        Takes in: number of folds
        
        Prints out RMSE score and stores the results in self.results
        """

        cv = KFold(n = self.X_train.shape[0], n_folds = n_folds)
        gbr = GradientBoostingRegressor(**self.params)
        self.med_error = []
        self.rmse_cv = []
        self.pct_error=[]
        self.results = {'pred': [],
                   'real': []}
        
        for train, test in cv:
            gbr.fit(self.X_train[train], self.y_train[train])
            pred = gbr.predict(self.X_train[test])
            predExp=np.exp(pred)
            testExp=np.exp(self.y_train[test])
            medError=median_absolute_error(predExp, testExp)
            percentError=np.median([np.fabs(p-t)/t for p,t in zip(predExp, testExp)])
            error = mean_squared_error(np.exp(pred), np.exp(self.y_train[test]))**0.5
            self.results['pred'] += list(pred)
            self.results['real'] += list(self.y_train[test])
            self.rmse_cv += [error]
            self.med_error+=[medError]
            self.pct_error+=[percentError]
        print 'Abs Median Error:', np.mean(self.med_error)
        print 'Abs Percent Error:', np.mean(self.pct_error)
        print 'Mean RMSE:', np.mean(self.rmse_cv)

    def kfold_cv_rand(self, n_folds = 3):
        """
        Takes in: number of folds
        
        Prints out RMSE score and stores the results in self.results
        """

        cv = KFold(n = self.X_train.shape[0], n_folds = n_folds)
        gbr = RandomForestRegressor(**self.params)
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
    def kfold_cv_rand(self, n_folds = 3):
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
        
    def plot_results(self, log=True):
        """
        Plots results from CV
        """
        if log:
            pMax=np.log(priceMax)
            pMin=np.log(priceMin)
        else:
            pMax=priceMax
            pMin=priceMin
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (12,10))

        ax.scatter(self.results['real'], self.results['pred'], color = (0.6,0.0,0.2),
                   label = 'Model Predictions',
                   s = 100, alpha = 0.4)
        ax.plot(np.arange(pMin, pMax),np.arange(pMin, pMax), color = 'black',
                   label = 'Perfect Prediction Line',
                   lw = 4, alpha = 0.5, ls = 'dashed')

        ax.set_xlabel('Log(Actual Price ($))',fontsize = 20)
        ax.set_ylabel('Log(Predicted Price ($))', fontsize = 20)
        ax.set_title('Results from KFold Cross-Validation', fontsize = 25)
        ax.set_xlim(pMin,pMax)
        ax.set_ylim(pMin,pMax)
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
        ax.set_xlim(0,50000)
        ax.set_ylim(0,50000)
        ax.legend(loc=2, fontsize = 16)
        ax.tick_params(labelsize =20)
def runPrediction(inDict={'breed':["Thoroughbred"],'age':[10],'inches':[66.],'gender':["Gelding"],'color':["Bay"], 'lnPrice':[1.0]}):
    df_test = all_data(pandasPath)
    df = df_test.copy()
    df = clean_col(df)
    lenTrain=len(df)
    testDf=pd.DataFrame.from_dict(inDict)
    total=pd.concat([df, testDf])
    total = total.reset_index().drop('index', axis = 1)
    total=encode(total)
    trainDf=total[:lenTrain]
    testDf=total[lenTrain:]
    model=Model(trainDf)
    gbr=model.makeModel()
    x_test=testDf.as_matrix(testDf.columns[:-1])
    pred=gbr.predict(x_test)
    return pred
