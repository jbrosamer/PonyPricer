'''

__file__

    scrape2sql.py

__description__

    This file uses the utilities in cglst_compact.py and atr_compact.py
    to scrape Craigslist and Autotrader, and then send the dataframes to
    a MySQL database

'''
import sys
import os
import datetime as dt
from collections import OrderedDict

import inspect
import pandas as pd
import numpy as np
import mysql.connector
import pickle, glob, cPickle
import math

import categories as cat

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno


class dbConnect():
    '''
    Class to help with context management for 'with' statements.

    http://www.webmasterwords.com/python-with-statement

    c = dbConnect(host = 'localhost', user = 'root',
                  passwd = 'default', db = 'nba_stats')
    with c:
        df.to_sql('nbadotcom', c.con, flavor = 'mysql', dtype = dtype)
    '''
    def __init__(self, host, user, passwd, db):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db
    def __enter__(self):
        self.con = mysql.connector.connect(host = self.host, user = self.user,
                                   passwd = self.passwd, db = self.db)
        self.cur = self.con.cursor()
    def __exit__(self, type, value, traceback):
        self.cur.close()
        self.con.close()
def pickleToDataframe(path="/Users/jbrosamer/PonyPricer/Batch/DressageId*.p"):
    df=pd.DataFrame()
    dfList=[]
    for f in glob.glob(path):
        print "opening",f
        p=pickle.load(open(f, 'rb'))
        dfList.append(p)
        print "N Rows %s %i"%(f, len(p))
    df = pd.concat(dfList, axis = 0)
    df = df.reset_index().drop('index', axis = 1)
    print "Total rows",len(df), "Line ", lineno()
    return df
def inches(row):
    inches=np.nan
    try:
         if "hh" in row["height"]:
            inches=float(row['height'].split('.')[0])*4.0+float(row['height'].split('.')[1][0])
         elif "Inch" in row['height']:
            inches=float(row['height'].replace(" Inches", ""))
    except:
        return np.nan
    if inches > 0:
        return inches
    return np.nan
def skillToStr(row):
    try:
        if pd.isnull(row['skills']):
            return ""
        return ",".join(row['skills'])[:50]
    except:
        return str(repr(row['skills'])).replace("'", "").replace("[", "").replace("]","")

def cleanBreed(row):
    if row['breed'] in cat.breeds:
        return row['breed']
    for b in cat.breeds:
        if b in row['breedStr']:
            return b
    return "Unknown"
def breedGroup(row):
    try:
        return cat.groupDict[row['breed']]
    except:
        return "Other"
def cleanGender(row):
    try:
        if row['gender'] in cat.genders:
            return str(row['gender'])
        if row['gender']=="Filly":
            return "Mare"
        if row['gender']=="Colt":
            return "Stallion"
        return ""
    except Exception as e:
        return "Gelding"
def lowerDesc(row):
    return (repr(row['desc'])).decode('utf-8').lower()

def hasKeyword(row, key):
    return key in row['desc']


def cleanDf(df):
    print "Total rows",len(df), "Line ", lineno()
    df=df[df.id >= 0]
    print "Total rows",len(df), "Line ", lineno()
    df=df[df.price > 0]
    print "Total rows",len(df), "Line ", lineno()
   # df=df[df.price < 100000]
    df=df[df.age >=0]
    print "Total rows",len(df), "Line ", lineno()
    df['inches']=df.apply(inches, axis=1)
    df['skills']=df.apply(skillToStr, axis=1)
    df['breed']=df.apply(cleanBreed, axis=1)
    df['gender']=df.apply(cleanGender, axis=1)
    df['desc']=df.apply(lowerDesc, axis=1)
    df['logprice']=df.apply(lambda x: math.log10(x['price']), axis=1)
    print "Total rows",len(df), "Line ", lineno()
    df['breedGroup']=df.apply(breedGroup, axis=1)
    for k in cat.keywords:
        print "Keyword",k
        df[k]=df.apply(lambda x: int(k in (x['desc'])), axis=1)
    for k in cat.skills:
        df[k]=df.apply(lambda x: int(k in (x['skills'])), axis=1)
    print "Total rows",len(df), "Line ", lineno()
    df['lenDesc']=df.apply(lambda x: len(x['desc']), axis=1)
    badCols=['breedStr', 'location', 'height']
    df=df.drop(badCols, axis=1)
    df = df.reset_index().drop('index', axis = 1)
    #print "df cols",df.columns()
    return df






def main():
    '''
    Takes class variables from cglst_compact.py and populates MySQL db.


    '''

    connect = dbConnect(host = 'localhost', user = 'root',
            passwd = 'jbrosamer', db = 'horses')
    keyword="All"
    #keyword="GreatLakes"
    with connect:
            path="/Users/jbrosamer/PonyPricer/BatchArea/%s*Ads.p"%keyword
            tablename = "%sAds"%keyword

            # Run Scraper
            df=pickleToDataframe(path)
            print "Total rows",len(df), "Line ", lineno()
            df=cleanDf(df)
            print "Total rows",len(df), "Line ", lineno()
            print "N rows",len(df)
            cPickle.dump(df, open("/Users/jbrosamer/PonyPricer/ConcatAds.p", 'wb'))
            df.to_csv("/Users/jbrosamer/PonyPricer/Batch/ConcatAds.csv")



if __name__ == "__main__":
    main()
