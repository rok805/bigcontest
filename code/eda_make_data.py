# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:36:07 2020

@author: User
"""


import pandas as pd
import utils
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from load_data import load_data
import datetime
import seaborn as sns
import pickle
import seaborn as sns
import matplotlib.font_manager as fm
import scipy.stats as stats
from collections import Counter

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

fm.get_fontconfig_fonts()
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='Malgun Gothic')

raw_path = utils.raw_path
processed_path = utils.processed_path
train_path = utils.train_path
fig_path = utils.fig_path


def plot_boxplot(list_, y_label, title, xtick_list=[''], 
                 save_path = '', save_option=False):
    
    """
    boxplot을 그리기 위한 함수
    """
    
    plt.figure(figsize=(24,16))
    plt.boxplot(list_)
    plt.xticks(range(1,len(xtick_list)+1),xtick_list)
    plt.ylabel(y_label)
    plt.title('%s boxplot'%(title), size=30)
    if save_option==True:
        plt.savefig(save_path)
    plt.show()

def plot_hist(list_, y_label, title, label_list='',
              save_path = '', save_option=False):
    
    """
    histogram을 그리기 위한 함수
    """
    
    plt.figure(figsize=(24,16))
    for i in range(len(list_)):
        element = list_[i]
        if label_list != '':
            sns.distplot(element, label=label_list[i])
    #plt.xticks([1],[''])
    plt.xlabel(y_label)
    plt.ylabel('scalied 빈도')
    plt.title('%s histogram'%(title), size=30)
    plt.legend()
    if save_option==True:
        plt.savefig(save_path)
    plt.show()


def make_str_month(x):
    month = str(x.month).zfill(2)
    return 'm%s'%(month)

def make_str_hour(x):
    hour = str(x.hour).zfill(2)
    return 'h%s'%(hour)

def make_str_weeknum(x):
    week = x.weekday()
    
    year= str(x.year).zfill(4)
    month = str(x.month).zfill(2)
    day = str(x.day).zfill(2)
    ymd = year+month+day
    
    if week >=5:
        if week == 6 or ymd in holiday_list:
            return 'week06'
        else:
            return 'week05'
    else:
        
        return str('week%s'%(str(week).zfill(2)))


def extract_holiday(year_, ymd_type=False):
    if year_ == 2019:
        holiday_list = ['20190101','20190204','20190205','20190206','20190301',
                        '20190505','20190506','20190512','20190606','20190815',
                        '20190912','20190913','20191003','20191009','20191225',
                        '20200101','20200606']
        
    if ymd_type == False:
        holiday_list = pd.to_datetime(holiday_list, format='%Y%m%d').tolist()
    return holiday_list

holiday_list = extract_holiday(2019, ymd_type=True)

def make_str_week(x):
    week = x.weekday()
    
    year= str(x.year).zfill(4)
    month = str(x.month).zfill(2)
    day = str(x.day).zfill(2)
    ymd = year+month+day
    
    if week <5:
        
        if ymd in holiday_list:
            return 'week01'
        else:
            return 'week00'
    
    else:
        
        return 'week01'

    
def make_str_holiday_weekend(x):
    week = x.weekday()
    
    year= str(x.year).zfill(4)
    month = str(x.month).zfill(2)
    day = str(x.day).zfill(2)
    ymd = year+month+day
    
    if week <5:
        if ymd in holiday_list:
            return 'week00'
        else:
            return 'week02'
    
    else:
        return 'week01'

def make_str_just_week(x):
    week = x.weekday()
    
    year= str(x.year).zfill(4)
    month = str(x.month).zfill(2)
    day = str(x.day).zfill(2)
    ymd = year+month+day
    
    if week <5:
        return 'week00'
    
    else:
        return 'week01'




def make_str_friday_week(x):
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


def make_str_season(x):
    month = x.month
    season = 0
    
    if month >=3 and month<6:
        season = str(0).zfill(2)
    elif month >=6 and month <9:
        season = str(1).zfill(2)
    elif month >=9 and month < 12:
        season = str(2).zfill(2)
    else:
        season = str(3).zfill(2)
    
    return 'season%s'%(season)


def make_str_min(x, interval=5):
    
    minute = x.minute
    minute_slot = 0
    quotient = int(60/interval)
    
    for i in range(quotient):
        if minute < (i+1)*interval:
            minute_slot = i
            break

    return '%sminute%s'%(interval, minute_slot)



def make_target_date_col(df, date_col):
    df['week'] = df[date_col].apply(lambda x: make_str_week(x))
    df['month'] = df[date_col].apply(lambda x: make_str_month(x))
    df['s_hour'] = df[date_col].apply(lambda x: make_str_hour(x))
    df['season'] = df[date_col].apply(lambda x: make_str_season(x))
    df['weeknum'] = df[date_col].apply(lambda x: make_str_weeknum(x))
    df['friday_week'] = df[date_col].apply(lambda x: make_str_friday_week(x))
    
    df['min5'] = df[date_col].apply(lambda x: make_str_min(x))
    df['s_hour_min5'] = ['%s_%s'%(df.loc[i,'s_hour'], df.loc[i,'min5']) for i in range(len(df))]
    df['s_hour_min5'] = df['s_hour_min5'].apply(lambda x:'%s:%s'%(x.split('_')[0][1:],  str(int(x.split('_')[1][7:])*5).zfill(2)))
    
    df['just_holiday_weekend'] = df[date_col].apply(lambda x: make_str_holiday_weekend(x))
    df['just_week'] = df[date_col].apply(lambda x: make_str_just_week(x))
    
    df['min10'] = df[date_col].apply(lambda x: make_str_min(x,10))
    df['s_hour_min10'] = ['%s_%s'%(df.loc[i,'s_hour'], df.loc[i,'min10']) for i in range(len(df))]
    df['s_hour_min10'] = df['s_hour_min10'].apply(lambda x:'%s:%s'%(x.split('_')[0][1:],  str(int(x.split('_')[1][8:])*10).zfill(2)))
    return df


def plot_hour_view(view_data, group_list, key_date,x_col, title, name_list='',
                   save_path='',save_option=False):
    
    g_view_data = view_data.groupby(group_list).mean().reset_index()
    g_view_data = g_view_data.sort_values('hour_order').reset_index(drop=True)
    unique_date_list = np.unique(g_view_data[key_date])
    plt.figure(figsize=(24,16))
    plt.title(title, size=30)
    if type(name_list) == str:
        name_list = unique_date_list
    for i in range(len(unique_date_list)):
        date = unique_date_list[i]
        name = name_list[i]
        temp_g_view_data = g_view_data[g_view_data[key_date]==date].reset_index(drop=True)
        plt.plot(range(len(temp_g_view_data)), temp_g_view_data['value'], label=name, marker='o')
        x = temp_g_view_data[x_col].apply(lambda x: '%s:00'%x[1:]).tolist()
        plt.xticks(range(len(temp_g_view_data)), x, rotation=30)
        plt.ylabel('시청률')
        plt.xlabel('시간')
    plt.legend()
    if save_option == True:
        plt.savefig(save_path+'%s.png'%(title))
    plt.show()



#%%
# 방송 데이터 EDA

def make_view_data():

    view_file_name = '2020 빅콘테스트 데이터분석분야-챔피언리그_시청률 데이터.xlsx'
    view = pd.read_excel(raw_path+view_file_name, header=1)
    
    view = view.drop(1440).reset_index(drop=True)
    view = view.drop(view.columns[-1], axis=1)
    
    hour_list = view['시간대']
    hour_list = hour_list.apply(lambda x: ' %s'%(x))
    hour_list = hour_list.values
    
    
    view_data = pd.DataFrame()
    for date_col in view.columns[1:]:
        value_list = view[date_col]
        date_list = pd.to_datetime(date_col + hour_list)
        temp_data = pd.DataFrame({'date':date_list, 'value':value_list})
        view_data = pd.concat([view_data, temp_data], axis=0)
    
    view_data = view_data.reset_index(drop=True)
    
    # 00:00 ~ 01:59 시간 전환
    temp_view_data = view_data[view_data['date'].apply(lambda x: x.hour)<2]
    temp_ind = temp_view_data.index
    
    view_data.loc[temp_ind, 'date'] = view_data.loc[temp_ind, 'date'] + datetime.timedelta(days=1)
    view_data.to_csv(processed_path+'view_data.csv', index=False)
    return 0

def plot_hour_minutes_view(temp_view_data, temp_name_list, stand_col,
                           add_title='', xlabel='', ylabel='', save_path='',
                           save_option=False):

    for k in range(len(temp_name_list)):
        target_var = 'week0%s'%(k)
        title_name = temp_name_list[k]
        t_temp_view_data = temp_view_data[temp_view_data[stand_col]==target_var].reset_index(drop=True)
        
        plt.figure(figsize=(24,16))
        plt.plot(range(len(t_temp_view_data)), t_temp_view_data['value'], alpha=0.3)
        for min_ in np.unique(t_temp_view_data['min']):
            target = t_temp_view_data[t_temp_view_data['min']==min_]
            plt.scatter(target.index, target['value'], label=min_)
            
        target = t_temp_view_data[t_temp_view_data['min']=='00']
        
        for i in target.index:
            plt.axvline(i, color='r', linestyle='--', alpha=0.5)
        
        title = ''
        if add_title == '':
            title = title_name
        else:
            title = add_title+title_name 
            
        plt.title(title, size=30)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(range(len(t_temp_view_data)), 
                   [i if ':00' in i else '' for i in t_temp_view_data['s_hour_min10']], 
                   fontsize=15)
        plt.legend()
        
        if save_option==True:
            plt.savefig(save_path+'.png')
        plt.show()
    


def analysis_view_rate(train):
    # 시청률 데이터 가공하여 생성
    make_view_data()
    
    
    train['방송일시n'] = train['방송일시n'].astype('datetime64[ns]')
    
    temp_target_data = train[['방송일시n','취급액','상품명','월','일','시간대','상품군']].copy()
    temp_target_data['방송일시n'] = temp_target_data['방송일시n'].astype('datetime64[ns]')
    
    # 10분 단위, 1시간 단위, 월간 단위 등을 분석하기 위해 각 column을 만듦
    temp_target_data = make_target_date_col(temp_target_data, '방송일시n')
    
    # 시청률 데이터 가져옴.
    t_view_data = pd.read_csv(processed_path+'view_data.csv', parse_dates=['date'])
    view_data = t_view_data.copy()
    
    
    view_data['date'] = view_data['date'].apply(lambda x: x - datetime.timedelta(days=1) if x.hour <2 else x)
    hour_key = pd.DataFrame({'s_hour':['h%s'%str(i).zfill(2) for i in [i for i in range(6,24)]+[0,1,2,3,4,5]],'hour_order':[i for i in range(24)]})
    
    
    view_data = make_target_date_col(view_data, 'date')
    view_data = pd.merge(view_data, hour_key, how='left', on='s_hour')
    
    # 시간별 시청률 그래프 그리기
    plot_hour_view(view_data, ['just_week','s_hour'], 'just_week','s_hour',
                   '주중별 시간별 평균시청률',['평일','주말'],
                   fig_path+'시청률/주중별 시간별 평균시청률', True)
    plot_hour_view(view_data, ['just_holiday_weekend','s_hour'],
                   'just_holiday_weekend','s_hour','주중 및 휴일별 시간별 평균시청률',
                   ['평일','주말','휴일'],
                   fig_path+'시청률/주중 및 휴일별 시간별 평균시청률', True)
    
    
    # 시간별_10분별 그래프 그리기
    
    temp_view_data = view_data.groupby(['just_week',
                                        's_hour_min10']).mean().reset_index()
    
    temp_view_data['min'] = temp_view_data['s_hour_min10'].apply(lambda x: x[3:])
    temp_view_data = temp_view_data.sort_values(['hour_order',
                                                 's_hour_min10']).reset_index(drop=True)

    temp_name_list = ['평일','주말']
    plot_hour_minutes_view(temp_view_data, temp_name_list, 'just_week',
                      '주중별_', '시간','시청률',fig_path+'시청률/',True)
    
    temp_view_data = view_data.groupby(['just_holiday_weekend',
                                        's_hour_min10']).mean().reset_index()
    temp_view_data['min'] = temp_view_data['s_hour_min10'].apply(lambda x: x[3:])
    temp_view_data = temp_view_data.sort_values(['hour_order',
                                                 's_hour_min10']).reset_index(drop=True)
    temp_name_list = ['평일','주말','휴일']
    plot_hour_minutes_view(temp_view_data, temp_name_list,'just_holiday_weekend', 
                      '주중 및 휴일별_', '시간','시청률',fig_path+'시청률/',True)


#%%
# 일시불 분석

# 무이자, 일시불 데이터 로드
def muiza_analysis(train):
    with open(processed_path+'train.csv_mu_il.pkl', 'rb') as f:
        mu_il_dict = pickle.load(f)
    
    
    ptype_list = np.unique(train['상품군']).tolist()
    price_list = []
    for key in mu_il_dict.keys():
        #value = mu_il_dict[key]
        target = train[train['상품명']==key].reset_index(drop=True)['판매단가']
        target_price = np.unique(target).tolist()
        price_list +=target_price
    
    #plot_box_hist(price_list, '판매단가','무이자 상품 판매단가')
    
    mu_mean = np.median(price_list)
    
    mu_df = train[train['일시불_무이자']==2].reset_index(drop=True)
    il_df = train[train['일시불_무이자']==1].reset_index(drop=True)

    # plot graph
    plot_hist([mu_df['취급액'], il_df['취급액']], 'scaled 빈도', 
              '일시불_무이자 취급액', ['무이자','일시불'],
              fig_path+'일시불_무이자/무이자_일시불_취급액_히스토그램.png', True)
    
    
    plot_boxplot([mu_df['취급액'],il_df['취급액']], '취급액', 
                 '일시불_무이자 취급액', ['무이자','일시불'],
                 fig_path+'일시불_무이자/무이자_일시불_취급액_박스플롯.png', True)
    
    # 무이자와 일시불에서 일치하는 변수 간 처리
    all_mu_list = [i for i in mu_il_dict.keys()]
    all_il_list = [i for i in mu_il_dict.values()]
    
    mu_dict = {}
    il_dict = {}
    
    price_dict = {}
    for ptype in ptype_list:
        mu_dict[ptype] = {'price':[], 'sales':[]}
        il_dict[ptype] = {'price':[], 'sales':[]}
        
        mu_price_list = []
        il_price_list = []
        
        mu_sales_list = []
        il_sales_list = []
        
        target_df = train[train['상품군']==ptype].reset_index(drop=True)
        
        effect_list = []
        ratio_price_list = []
        price_list = []
        discount_price_list=[]
        name_list = []
        for mu_product in all_mu_list:
            target = target_df[target_df['상품명_n']==mu_product].sort_values('방송일시')
            target_sales = target['취급액'].tolist()
            
            if mu_product not in mu_il_dict.keys() or len(target)==0:
                continue
            il_product = mu_il_dict[mu_product]
            il_target = target_df[target_df['상품명_n']==il_product].sort_values('방송일시')
            il_target_sales = il_target['취급액'].tolist()
            
            temp_final_df = target[['방송일시','취급액','판매단가']]
            temp_final_df.columns = ['방송일시','무이자_취급액','무이자_판매단가']
            t_il_target = il_target[['방송일시','취급액','판매단가','상품명']].copy()
            t_il_target.columns = ['방송일시','일시불_취급액','일시불_판매단가','상품명']
            temp_final_df = pd.merge(temp_final_df, t_il_target,on='방송일시',how='inner')
            temp_final_df['일시불_할인액'] = temp_final_df['무이자_판매단가'] - temp_final_df['일시불_판매단가']
            temp_final_df['일시불_할인율'] = (temp_final_df['무이자_판매단가'] - temp_final_df['일시불_판매단가'])/temp_final_df['무이자_판매단가']
            
            temp_final_df = temp_final_df[temp_final_df['무이자_취급액']!=0]
            
            temp_final_df['효과'] = temp_final_df['일시불_취급액']/temp_final_df['무이자_취급액']
            discount_values = np.unique(temp_final_df['일시불_할인액'])
            for discount_v in discount_values:
                
                t_temp_final_df = temp_final_df[temp_final_df['일시불_할인액']==discount_v]
                discount_ratio = np.unique(t_temp_final_df['일시불_할인율'])[0]
                price = np.unique(t_temp_final_df['일시불_판매단가'])[0]
                effect = t_temp_final_df['효과'].tolist()
                name = np.unique(t_temp_final_df['상품명'])[0]
                
                effect_list+=effect
                ratio_price_list+= [discount_ratio]*len(effect)
                price_list+=[price]*len(effect)
                discount_price_list+=[discount_v]*len(effect)
                name_list+=[name]*len(effect)
        price_dict[ptype] = {'effect':effect_list,'discount_ratio':ratio_price_list,
                             'discount_price':discount_price_list, 'price':price_list,
                             'name':name_list}
    
    effect_list = []
    ratio_price_list = []
    price_list = []
    discount_price_list=[]
    name_list = []
    for key in price_dict.keys():
        effect_list+= price_dict[key]['effect']
        ratio_price_list+= price_dict[key]['discount_ratio']
        price_list+= price_dict[key]['discount_price']
        discount_price_list+= price_dict[key]['price']
        name_list += price_dict[key]['name']
    
    price_df = pd.DataFrame({'effect':effect_list, 'discount_ratio':ratio_price_list,
                            'discount_price':price_list,
                            'price':discount_price_list,
                            'name':name_list})
    
    corr = price_df.corr()['effect']['discount_ratio']
    print('effect와 discount_ratio의 상관관계 : %.3f'%(corr))
    
    mean_v = np.mean(price_df['price'])
    return 0


def normalized(value):
    value = (value-min(value))/(max(value)-min(value))
    return value

def plot_graph(target_list,x, x_label, y_label, title, label_list=[''],
               smoothing = 20, save_path='', save_option=False):
    plt.figure(figsize=(24,16))
    for i in range(len(target_list)):
        target = target_list[i]
        label = label_list[i]
        plt.plot(range(len(target)), target.rolling(smoothing).mean(), label=label)
    plt.title(title, size=30)
    plt.xlabel(x_label, size=15)
    plt.ylabel(y_label, size=15)
    plt.xticks(range(len(x)), x, rotation=30)
    plt.legend()
    if save_option==True:
        plt.savefig(save_path)
    plt.show()
            
def eda_transport(df, save_path, group_option):
    df = bind_y_x_var(df, '대중교통', group_option)   
    value = normalized(df['취급액'])
    x = ['%s-%s-%s'%(i[:4],i[4:6],i[6:8]) if i[6:8] =='01' else '' for i in df['ymd']]
    target = normalized(df['대중교통'])
    plot_graph([value,target], x, '날짜', '값', '취급액과 대중교통 그래프', 
               label_list=['취급액','대중교통'],  
               save_path=save_path, 
               save_option=True)
    

    
def eda_dust(df, save_path, group_option):
    df = bind_y_x_var(df, 'PM10', group_option)   
    value = normalized(df['취급액'])
    x = ['%s-%s-%s'%(i[:4],i[4:6],i[6:8]) if i[6:8] =='01' else '' for i in df['ymd']]
    target = normalized(df['PM10'])
    plot_graph([value,target], x, '날짜', '값', '취급액과 PM10 그래프', 
               label_list=['취급액','PM10'],  
               save_path=save_path, 
               save_option=True)

    

def bind_y_x_var(df, col, group_option):
    df = df[['방송일시',col,'취급액']]
    df['방송일시'] = df['방송일시'].astype('datetime64[ns]')
    df['ymd'] = df['방송일시'].apply(lambda x: utils.make_ymd(x))
    
    if group_option == 'mean':
        df =df.groupby('ymd').mean().reset_index()
    
    elif group_option == 'sum':
        df =df.groupby('ymd').sum().reset_index()
    
    return df

    
def main():
    train = load_data('train.csv', '',False)
    
    # 방송 데이터 EDA
    view_data(train)
    
    # 이자,무이자 EDA
    muiza_analysis(train)
    
    # 대중교통
    eda_transport(train,fig_path+'대중교통/취급액과 대중교통 그래프', 'sum')
    
    # 미세먼지
    eda_dust(train,fig_path+'미세먼지/취급액과 미세먼지 그래프', 'mean')
    
    # 
    