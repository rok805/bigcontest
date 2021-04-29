# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:42:22 2020

@author: User
"""


from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import utils

raw_path = utils.raw_path
train_path = utils.train_path
processed_path = utils.processed_path
fig_path = utils.fig_path


# Create processed air polution data by time
def air_pol():
    # Load air polution data for train data
    all_files = glob.glob(raw_path + 'air_polution/*.xlsx')
    train_air = pd.DataFrame()
    for f in all_files:
        each = pd.read_excel(f)
        train_air = pd.concat([train_air, each])

    bytime = train_air.groupby('측정일시').mean()
    
    date_ = []
    
    for i in bytime.index.tolist():
        str_i = str(i)
        date_.append(datetime(int(str_i[:4]),
                              int(str_i[4:6]),
                              int(str_i[6:8])))

    bytime['time'] = date_
    bytime = bytime.groupby('time').mean()
    bytime = bytime[['PM10', 'O3']]
    
    return bytime


# Merge with sales to compare
def make_traff_df(t_df, air_df):
    t_df['방송일시'] = t_df['방송일시'].apply(lambda x: datetime(int(x[:4]),
                                                                 int(x[5:7]),
                                                                 int(x[8:10])))
    t_df = t_df[['방송일시', '취급액', '방송set']]
    uniq_day = list(np.unique(t_df['방송일시']))
    df_day = []

    for u in uniq_day:
        df_day.append(t_df[t_df['방송일시'] == u])

    day_365 = pd.DataFrame()

    for i in df_day:
        df_set = i.groupby('방송set').sum()
        df_mean = i.groupby('방송일시').mean()
        df_mean.reset_index(inplace=True)
        day_365 = pd.concat([day_365, df_mean[['방송일시', '취급액']]])

    day_365['PM10'] = air_df['PM10'].tolist()
    day_365['O3'] = air_df['O3'].tolist()
    day_365.set_index('방송일시', inplace=True)
    
    return day_365

def make_air_pollution_data():
    # Load train data to compare with are data

    
    train_test_air = air_pol()
    train_test_air['time'] = train_test_air.index
    train_test_air = train_test_air.reset_index(drop=True)
    
    ind = train_test_air[train_test_air['time']=='2019-12-31'].index
    
    train_test_air = train_test_air.append({'time':datetime(2020,1,1),
                                            'PM10':train_test_air.loc[ind,'PM10'],
                                            'PM10':train_test_air.loc[ind,'O3']},
                                           ignore_index=True)
    
    train_test_air = train_test_air.to_csv(processed_path+'air_train_test.csv', index=False)