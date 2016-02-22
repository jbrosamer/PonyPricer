import re, glob, pickle, os
import cPickle

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import  stats
from categories import keywords
from categories import skills


skillsCol=['dressage', 'hunt', 'jump', 'event', 'prospect', 'import']
baseCols=['age', 'gender', 'inches', 'color', 'breed']
baseCols+=['logprice']
allCols=['age', 'gender', 'inches', 'color', 'breed']+skills
allCols+=['logprice']
txtCols=['desc', 'logprice']
lblColumns=['breed', 'color', 'gender', 'breedGroup']
final_cols=baseCols
# Path of data

fromPickle=True
priceMin=1000

priceMax=100000
pandasPath="/Users/jbrosamer/PonyPricer/ConcatAds.p"
if final_cols==allCols:
    modelPath="%s/ModelAllCols/"%os.path.dirname(os.path.abspath(__file__))
else:
    modelPath="%s/ModelBaseCols"%os.path.dirname(os.path.abspath(__file__))


def pow10(npArray):
    """
        Slightly faster way to take 10^ for reversing log
    """
    return np.array([10**x for x in npArray])
    
def all_data(path=pandasPath):
    """
    Takes in: with wildcarding of dataframes stored in .p
    
    Returns: a dataframe of all ads
    """
    df=pickle.load(open(path, 'rb'))    
    return df
def cleanGender(row):
    """
        Make sure dataframe has valid gender
    """
    if "Mare" in row['gender'] or "Filly" in row['gender']:
        return 1
    return 0




def clean_col(df):
    """
        Make sure all columns are valid, remove duplicates and unclean data
    """
    df=df.drop_duplicates(subset=['id'])
    df=df[(df['age']>0) & (df['price']>=priceMin) &  (df['price']<=priceMax) & (df['inches']>50) & (df['gender'] != '') & (df['breed'] != "Unknown")]
    df = df.reset_index().drop('index', axis = 1)
    return df





def encode(df, dump=fromPickle):
    """
    Takes in: dataframe from clean_col
    
    Returns: a dataframe that LabelEncodes the categorical variables
    """
    encoders=dict()
    for col in lblColumns:
        if col not in final_cols:
            continue
        le = LabelEncoder()
        if dump:
            fName="%s/%s.npy"%(modelPath,col)
            if os.path.isfile(fName):
                le.classes_=np.load(fName)
            else:
                le.fit(df[col])
                np.save(fName, le.classes_)
        else:
            le.fit(df[col])
        encoders[col]=le
        df[col] = le.transform(df[col])
    # Order columns with logprice as the last column
    df = df[final_cols]
    df = df.reset_index().drop('index', axis = 1)
    return df

class TxtFeatures():
    """
    Small class to do tf-idf feature identification
    """
    def __init__(self, df=None):
        """
        Load data and intialize object
        """
        if df is None:
            df=all_data()
        self.df=clean_col(df)
        self.df=self.df[txtCols]
        self.df = self.df.reset_index().drop('index', axis = 1)
        self.test_size = 0.1
    def split(self):
        """
        Split for testing
        """
        np.random.seed(1)
        self.df = self.df.reindex(np.random.permutation(self.df.index))
        self.df = self.df.reset_index().drop('index', axis = 1)
        X = self.df.as_matrix(self.df.columns[:-1])
        y = self.df.as_matrix(['logprice'])[:,0]
        X_train, X_test, y_train, y_test = train_test_split(
                                                X, y,
                                                test_size=self.test_size,
                                                )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.gbr=None
        
    def fit(self):
        """
        Fit tfidf vectorizer to df
        """
        self.split()
        vectorizer=TfidfVectorizer(stop_words='english')
        vX_train = vectorizer.fit_transform(self.X_train)
        print("n_samples: %d, n_features: %d" % vX_train.shape)
        vX_test = vectorizer.transform(self.X_test)
        print("n_samples: %d, n_features: %d" % vX_test.shape)
        feature_names = vectorizer.get_feature_names()
        




class Model():
    
    def __init__(self, df=None, test_size = 0.3, params={'n_estimators':1000, 'max_depth': 2, 'min_samples_split':1,
 'min_samples_leaf':2 }):
        if df is None:
            df=all_data()
        self.df = df
        self.params = params
        self.test_size = 0.1
        self.X_train = self.X_test = self.df.as_matrix(self.df.columns[:-1])
        self.y_train = self.y_test  = self.df.as_matrix(['logprice'])[:,0]
        
    def split(self):
        np.random.seed(1)
        self.df = self.df.reindex(np.random.permutation(self.df.index))
        self.df = self.df.reset_index().drop('index', axis = 1)
        X = self.df.as_matrix(self.df.columns[:-1])
        y = self.df.as_matrix(['logprice'])[:,0]
        X_train, X_test, y_train, y_test = train_test_split(
                                                X, y,
                                                test_size=self.test_size,
                                                )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.gbr=None

    def makeModel(self, dump=fromPickle):
        """
            fit GBR model with all data
        """
        gbr = GradientBoostingRegressor(**self.params)
        self.X=self.df.as_matrix(self.df.columns[:-1])
        self.Y=self.df.as_matrix(['logprice'])[:,0]


        gbr.fit(self.X, self.Y)
        self.gbr=gbr
        return gbr


        
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
        self.r2=[]
        self.results = {'pred': [],
                   'real': []}
        
        for train, test in cv:
            gbr.fit(self.X_train[train], self.y_train[train])
            pred = gbr.predict(self.X_train[test])
            print "Score", gbr.score(self.X_train[test], self.y_train[test])
            predExp=np.power(10, pred)
            testExp=np.power(10, self.y_train[test])
            medError=median_absolute_error(predExp, testExp)
            percentError=np.median([np.fabs(p-t)/t for p,t in zip(predExp, testExp)])
            error = mean_squared_error(np.power(10, pred), np.power(10, self.y_train[test]))**0.5
            self.results['pred'] += list(pred)
            self.results['real'] += list(self.y_train[test])
            self.rmse_cv += [error]
            self.med_error+=[medError]
            self.pct_error+=[percentError]
            self.r2+=[r2_score(self.y_train[test], pred)]
        print 'Abs Median Error:', np.mean(self.med_error)
        print 'Abs Percent Error:', np.mean(self.pct_error)
        print 'Mean RMSE:', np.mean(self.rmse_cv)
        print "R2",np.mean(self.r2)

    def cross_val_cols(self, n_folds = 3):
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
            dfFeatures+=[unencode(pd.DataFrame(columns=final_cols[:-1], data=self.X_train[test]))]
            pred = gbr.predict(self.X_train[test])
            medError=median_absolute_error(predExp, testExp)
            percentError=np.median([np.fabs(p-t)/t for p,t in zip(predExp, testExp)])
            error = mean_squared_error(np.power(pred, 10), np.power(self.y_train[test], 10))**0.5
            self.inFeatures=(self.X_train[test])
            self.results['pred'] += list(predExp)
            self.results['real'] += list(testExp)
            self.rmse_cv += [error]
            self.med_error+=[medError]
            self.pct_error+=[percentError]
        print 'Abs Median Error:', np.mean(self.med_error)
        print 'Abs Percent Error:', np.mean(self.pct_error)
        print 'Mean RMSE:', np.mean(self.rmse_cv)
        self.valDf=pd.DataFrame.concat(dfFeatures)
        self.valDf= self.valDf.reset_index().drop('index', axis = 1)
        self.valDf['pred']=self.results['pred']
        self.valDf['real']=self.results['real']
        return self.valDf

    def kfold_cv_rand(self, n_folds = 3):
        """
        Takes in: number of folds
        
        Prints out RMSE score and stores the results in self.results
        """
        cv = KFold(n = self.X_train.shape[0], n_folds = n_folds)
        gbr = RandomForestRegressor(**self.params)
        self.med_error = []
        self.rmse_cv = []
        self.pct_error=[]
        self.r2=[]
        self.results = {'pred': [],
                   'real': []}
        
        for train, test in cv:
            print "Starting fit"
            gbr.fit(self.X_train[train], self.y_train[train])
            pred = gbr.predict(self.X_train[test])
            predExp=np.power(pred, 10)
            testExp=np.power(self.y_train[test], 10)
            medError=median_absolute_error(predExp, testExp)
            percentError=np.median([np.fabs(p-t)/t for p,t in zip(predExp, testExp)])
            error = mean_squared_error(np.power(pred, 10), np.power(self.y_train[test], 10))**0.5
            self.results['pred'] += list(pred)
            self.results['real'] += list(self.y_train[test])
            self.rmse_cv += [error]
            self.med_error+=[medError]
            self.pct_error+=[percentError]
            self.r2+=[r2_score(self.y_train[test], pred)]
        print 'Abs Median Error:', np.mean(self.med_error)
        print 'Abs Percent Error:', np.mean(self.pct_error)
        print 'Mean RMSE:', np.mean(self.rmse_cv)
        print "R2",np.mean(self.r2)
        
    def plot_results(self, log=True):
        """
        Plots results from CV
        Slow right now but unsure why!
        """
        pMax=priceMax*5
        pMin=priceMin/5
        print "Starting"
        if log:
            if not self.results.has_key('pred10'):
                self.results['pred10']=pow10(self.results['pred'])
            y=self.results['pred10']
            if not self.results.has_key('real10'):
                self.results['real10']=pow10(self.results['real'])
            x=self.results['real10']
        else:
            
            x=self.results['real']
            y=self.results['pred']
        plt.style.use('ggplot')
        print "going to plot"
        fig, ax = plt.subplots(figsize = (12,10))
        ax.set(xscale="log", yscale="log")
        ax.set_xlim(pMin,pMax)
        ax.set_ylim(pMin,pMax)

        ax.scatter(x=x, y=y, color = (0.6,0.0,0.2),
                   label = 'Model Predictions',
                   s = 100, alpha = 0.05)

        ax.plot(np.arange(pMin, pMax*100),np.arange(pMin, pMax*100), color = 'black',
                   label = 'Perfect Prediction Line',
                   lw = 4, alpha = 0.5, ls = 'dashed')
        ax.set_xlabel('Actual Price [$]', fontsize = 40)
        ax.set_ylabel('Predicted Price [$]', fontsize = 40)
       # ax.set_title('Results from KFold Cross-Validation', fontsize = 40)
        
        
        ax.legend(loc=2, fontsize=30)
        ax.tick_params(labelsize =20)
        plt.show()

    def plotFeatures(self, nFeat=8):
        importances = self.gbr.feature_importances_
        # std = np.std([tree.feature_importances_ for tree in gbr.estimators_],
        #              axis=0)
        indices = np.argsort(importances)[::-1]
        self.importances=importances
        self.indices=indices
        print("Feature ranking:")
        outfile=open("Features.txt", 'wb')

        for f in range(self.X.shape[1]):
            outfile.write("%s,%f\n"%(final_cols[indices[f]], importances[indices[f]]))
            print("%s %d. feature %d (%f)" % (final_cols[indices[f]], f + 1, indices[f], importances[indices[f]]))
        outfile.close()
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        xvals=range(self.X.shape[1])
        plt.bar(range(self.X.shape[1]), importances[indices],
               color="r", align="center")
        self.featNames=["%s"%(final_cols[x]) for x in indices]

        plt.xticks(range(self.X.shape[1]), self.featNames)
        plt.xlim([-1, self.X.shape[1]])
        plt.show()

    def plotPartial(self, nFeat=2):
        features = self.indices[:nFeat]
        print "features",features
        featNames=final_cols
        print "FeatureNames",featNames
        fig, axs = plot_partial_dependence(self.gbr, self.X, features, feature_names=featNames)

        print('_' * 80)
        print('Custom 3d plot via ``partial_dependence``')
        print
        fig = plt.figure()
        plt.show()


    
    def validate(self, pickle=True, cv=0):
        """
        Validate Model on Test set
        """
        if cv>0:
            self.split()
        else:
            self.X_train=self.X_test = self.df.as_matrix(self.df.columns[:-1])
            self.y_train=self.y_test = self.df.as_matrix(['logprice'])[:,0]
        if pickle:
            gbr=cPickle.load(open("%s/Model.pkl"%modelPath, 'rb'))
        else:
            gbr = GradientBoostingRegressor(**self.params)
            gbr.fit(self.X_train, self.y_train)
        self.results = {'pred': [],
                   'real': []}
        self.results['pred'] = gbr.predict(self.X_test)
        self.results['real'] = self.y_train
        self.results['pred10']=pow10(self.results['pred'])
        self.results['real10']=pow10(self.results['real'])
        print "Score ",r2_score(self.y_train, self.results['pred'])
def predDataframe():
    df = all_data(pandasPath)
    df = clean_col(df)
    lblDf=df.copy()
    lblDf=lblDf[final_cols]
    df=encode(df)
    model=Model(df)
    gbr=model.makeModel()
    pred=gbr.predict(model.X)
    lblDf['real']=np.power(df['logprice'])
    lblDf['pred']=np.power(pred, 10)
    lblDf['diff']=pd.Series(np.fabs(lblDf['pred']-lblDf['real']))
    lblDf['perDiff']=pd.Series(np.fabs(lblDf['pred']-lblDf['real'])/lblDf['real'])

    return lblDf

def predCVDataframe():
    df = all_data(pandasPath)
    df = clean_col(df)
    lblDf=df.copy()
    lblDf=lblDf[final_cols]
    df=encode(df)
    model=Model(df)
    gbr=model.makeModel()
    pred=gbr.predict(model.X)
    lblDf['real']=np.power(df['logprice'], 10)
    lblDf['pred']=np.power(pred, 10)
    lblDf['diff']=pd.Series(np.fabs(lblDf['pred']-lblDf['real']))
    lblDf['perDiff']=pd.Series(np.fabs(lblDf['pred']-lblDf['real'])/lblDf['real'])

    return lblDf



def runPrediction(inDict={'breed':["Thoroughbred"],'age':[10],'inches':[66.],'gender':["Gelding"],'color':["Bay"], 'logprice':[1.0]}, conInt=None):
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
    gbr=model.makeModel(dump=fromPickle)
    x_test=testDf.as_matrix(testDf.columns[:-1])
    pred=gbr.predict(x_test)
    return pred

def predForWeb(inDict={'breed':["Thoroughbred"],'age':[10],'inches':[66.],'gender':["Gelding"],'color':["Bay"], 'logprice':[1.0]}, conInt=None):
    gbr=cPickle.load(open("%s/Model.pkl"%modelPath, 'rb'))
    testDf=pd.DataFrame.from_dict(inDict)
    testDf=encode(testDf, True)
    x_test=testDf.as_matrix(testDf.columns[:-1])
    pred=gbr.predict(x_test)
    return pred

def saveModels():
    df= all_data(pandasPath)
    df = clean_col(df)
    df=encode(df, dump=True)
    model=Model(df)
    gbr=model.makeModel()
    print "Dumping","%sModel.pkl"%modelPath
    cPickle.dump(gbr, open("%s/Model.pkl"%modelPath, 'wb'))
    return gbr




