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

import pandas as pd
import numpy as np
import mysql.connector
import pickle, glob

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
    df = pd.concat(dfList, axis = 0)
    df = df.reset_index().drop('index', axis = 1)
    return df
def inches(row):
    try:
         if "hh" in row["height"]:
            return float(row['height'].split('.')[0])*4.0+float(row['height'].split('.')[1][0])
         if "Inch" in row['height']:
            return float(row['height'].replace(" Inches", ""))
    except:
        return 0
def skillToStr(row):
    try:
        if pd.isnull(row['skills']):
            return ""
        return ",".join(row['skills'])
    except:
        return ""


def cleanDf(df):

    df=df[df.id > 0]
    df=df[df.price > 0]
    df=df[df.price < 100000]
    df=df[df.age >0]
    df['inches']=df.apply(inches, axis=1)
    df['skills']=df.apply(skillToStr, axis=1)
    # allCols=[u'id', u'breed', u'breedStr', u'price', u'color', u'location', u'age',
    #    u'zip', u'height', u'temp', u'warmblood', u'sold', u'soldhere',
    #    u'forsale', u'forlease', u'registered', u'skills', u'desc', u'inches']
    badCols=['breedStr', 'desc', 'location', 'height']
    # badCols=[u'breed', u'breedStr', u'price', u'color', u'location', u'age',
    #    u'zip', u'height', u'temp', u'warmblood', u'sold', u'soldhere',
    #    u'forsale', u'forlease', u'registered', u'skills', u'desc', u'inches']
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
    with connect:
            tablename = "dressageAds"

            # Run Scraper
            df=pickleToDataframe()
            df=cleanDf(df)


            # Make sure dtypes fit for MySQL db
            dtype = {}
            for i in range(len(df.columns)):
                if df.columns[i] in ['warmblood', 'sold', 'soldhere', 'forlease',
                                     'forlease']:
                    dtype[df.columns[i]] = 'BOOLEAN'
                elif df.columns[i] in ['id', 'temp']:
                    dtype[df.columns[i]] = 'INTEGER'
                elif df.columns[i] in ['price', 'height', 'age']:
                    dtype[df.columns[i]] = 'REAL'
                else:
                    dtype[df.columns[i]] = 'TEXT'
            print dtype
            x = pd.DataFrame({'x': [1, 2, 3], 'y': [3, 4, 5]})
            df.to_sql(name = tablename, con = connect.con,
                      flavor = 'mysql', if_exists='replace')

            # Send to MySQL database, if table exists, continue to next
            try:
                df.to_sql(name = tablename, con = connect.con,
                      flavor = 'mysql', dtype = dtype, if_exists='replace')
                print tablename
                print
            except Exception as e:
                print "Exception ",e


if __name__ == "__main__":
    main()
