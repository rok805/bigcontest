# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:42:25 2020

@author: User
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
"""
import matplotlib 
import matplotlib.font_manager as fm



font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

fm.get_fontconfig_fonts()
font_location = 'C:/Windows/Fonts/NanumGothic.ttf' # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)
"""
data_path = os.path.abspath('../data/')
save_path = os.path.abspath('../save/')
train_path = os.path.join(data_path,'train/')
test_path = os.path.join(data_path,'test/')
raw_path = os.path.join(data_path,'raw/')
other_path = os.path.join(data_path,'other/')

processed_path = os.path.join(data_path,'processed/')
output_path = os.path.abspath('../output/')
fig_path = os.path.join(output_path, 'fig/')
model_path = os.path.join(output_path, 'model/')
config_path = os.path.join(os.getcwd(),'config/')


def print_indicator(list_, name='pattern', box_option=True):
    if type(list_) == list:
        list_ = np.array(list_)
    
    list_ = np.round(list_,2)
        
    print('평균 : %s'%np.round(np.mean(list_),2))
    print('표준편차 : %s'%np.round(np.std(list_),2))
    print('중간값 : %s'%np.median(list_))
    print('최대값 : %s'%max(list_))
    print('최솟값 : %s'%min(list_))
    print('제1사분위값 : %s'%np.percentile(list_,25))
    print('제3사분위값 : %s'%np.percentile(list_,75))
    
    if box_option == True:
        plt.figure(figsize=(24,8))
        plt.title('pattern')
        plt.boxplot(list_)    
        plt.show()
        plt.close()
        
        
def make_same_columns(df_, target_col_, col_name_,date_col='방송일시', same_drop=True):
    
    df_ = df_.sort_values([target_col_,date_col]).reset_index(drop=True)
    
    past_date = df_.loc[0,date_col]
    past_code = df_.loc[0,target_col_]
    max_time = max(df_['노출(분)'])
    
    same = 0
    same_list = []
    
    for i in range(len(df_)):
        cur_date = df_.loc[i,date_col]
        cur_code = df_.loc[i,target_col_]
        diff = ((cur_date - past_date).total_seconds())/60
        
        if past_code == cur_code and diff <= max_time:
            same_list.append(same)
            
        else:
            same +=1
            same_list.append(same)
        
        past_date = cur_date
        past_code = cur_code
        
        
    df_['same'] = same_list
    
    if same_drop == True:
        df_ = df_.drop('same',axis=1)
    return df_



def make_hm(x):
    hour = str(x.hour).zfill(2)
    minute = str(x.minute).zfill(2)
    result = hour+':'+minute
    
    return result

def make_mh(x):
    month = str(x.month).zfill(2)
    hour = str(x.hour).zfill(2)
    result = month+'/'+hour
    
    return result


def make_md(x):
    month = str(x.month).zfill(2)
    day = str(x.day).zfill(2)
    result = month+'/'+day
    
    return result

def make_ymd(x):
    year = str(x.year).zfill(4)
    month = str(x.month).zfill(2)
    day = str(x.day).zfill(2)
    result = year+month+day
    
    return result


def extract_holiday(year_, ymd_type=False):
    if year_ == 2019:
        holiday_list = ['20190101','20190204','20190205','20190206','20190301','20190506','20190606',
                    '20190815','20190912','20190913','20191003','20191009','20191225','20200101','20200606']
    if ymd_type == False:
        holiday_list = pd.to_datetime(holiday_list, format='%Y%m%d').tolist()
    return holiday_list

holiday_list = extract_holiday(2019, ymd_type=True)

def make_week(x):
    week = x.weekday()
    
    year= str(x.year).zfill(4)
    month = str(x.month).zfill(2)
    day = str(x.day).zfill(2)
    ymd = year+month+day
    
    if week <5:
        
        if ymd in holiday_list:
            return '1'
        else:
            return '0'
    
    else:
        
        return '1'


def distinct_weeknum(x):
    if type(x) == str:
        x = int(x)
    if x == 0:
        return 'MON'
    elif x == 1:
        return 'TUE'
    elif x == 2:
        return 'WED'
    elif x == 3:
        return 'THU'
    elif x == 4:
        return 'FRI'
    elif x == 5:
        return 'SAT'
    elif x == 6:
        return 'SUN'


def make_friday_week(x):
    week = x.weekday()
    hour = x.hour
    
    year= str(x.year).zfill(4)
    month = str(x.month).zfill(2)
    day = str(x.day).zfill(2)
    
    ymd = year+month+day
    
    cur_date = pd.to_datetime('%s-%s-%s'%(year, month, day))
    after_date = cur_date + datetime.timedelta(days=1)
    aft_year = str(after_date.year).zfill(4)
    aft_month = str(after_date.month).zfill(2)
    aft_day = str(after_date.day).zfill(2)
    
    aft_ymd = aft_year+aft_month+aft_day
    
    if week <4:
        
        if ymd in holiday_list:
            return 'week01'
        else:
            if aft_ymd in holiday_list:
                
                if hour>= 22:
                    return 'week01'
                else:
                    return 'week00'
            else:
                    return 'week00'
                
    elif week == 4:
        if hour >= 22:
            return 'week01'
        else:
            return 'week00'
        
    else:
        
        return 'week01'

def make_s_hour(x):
    hour = str(x.hour).zfill(2)
    return 'h%s'%(hour)





def make_same_columns_price(df_, target_col_, col_name_,date_col='방송일시', same_drop=True):
    
    df_ = df_.sort_values([target_col_,date_col]).reset_index(drop=True)
    if '%s총취급액'%(col_name_) in df_.columns:
        df_ = df_.drop('%s총취급액'%(col_name_), axis=1)
    if '%s평균취급액'%(col_name_) in df_.columns:
        df_ = df_.drop('%s평균취급액'%(col_name_), axis=1)
    
    
    past_date = df_.loc[0,date_col]
    past_code = df_.loc[0,target_col_]
    max_time = max(df_['노출(분)'])
    
    same = 0
    same_list = []
    
    for i in range(len(df_)):
        cur_date = df_.loc[i,date_col]
        cur_code = df_.loc[i,target_col_]
        diff = ((cur_date - past_date).total_seconds())/60
        
        if past_code == cur_code and diff <= max_time:
            same_list.append(same)
            
        else:
            same +=1
            same_list.append(same)
        
        past_date = cur_date
        past_code = cur_code
        
        
    df_['same'] = same_list
    
    g_s_train = df_.groupby(['same']).sum().reset_index()[['same','취급액']]
    g_s_train.columns = ['same','%s총취급액'%(col_name_)]
    g_m_train = df_.groupby(['same']).mean().reset_index()[['same','취급액']]
    g_m_train.columns = ['same','%s평균취급액'%(col_name_)]
    
    df_ = pd.merge(df_, g_s_train, on='same', how='left')
    df_ = pd.merge(df_, g_m_train, on='same', how='left')
    if same_drop == True:
        df_ = df_.drop('same',axis=1)
    return df_