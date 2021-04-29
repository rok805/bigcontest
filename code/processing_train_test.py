# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:50:44 2020

@author: User
"""
import pandas as pd
import numpy as np
import os
import pickle
import utils
import matplotlib.pyplot as plt
import datetime
import time
import glob
import pdb
import warnings
from make_air_pollution import make_air_pollution_data

#%%


def processing_nocul_time(df_, type_='train'):
    
    # 1. 노출분 편집
    product_list = np.unique(df_['상품명']).tolist()
    df_ = df_.sort_values(['상품명','방송일시']).reset_index(drop=True)
    df_['origin_노출(분)'] = 0
    
    df_['방송일시'] =df_['방송일시'].astype('datetime64[ns]')
    
    max_time = max(df_['노출(분)'])
    
    df_ = utils.make_same_columns(df_, '상품명', '상품별', date_col='방송일시', same_drop=False)
    same_list = np.unique(df_['same']).tolist()
    
    except_name_list = ['아미니 비노테라 오일 워시(구성1)','아미니 비노테라 오일 워시(구성2)']
    except_mother_list = [100226, 100711]
    for same in same_list:
        t_df = df_[df_['same']==same]
        ind_list = t_df.index
        t_df = t_df.reset_index(drop=True)
        if len(t_df) > 1:
            time = ((t_df.loc[1,'방송일시'] - t_df.loc[0,'방송일시']).total_seconds())/60
        else: 
            
            time = t_df.loc[0,'노출(분)']
        
        df_.loc[ind_list, 'origin_노출(분)'] = time
        if max_time*2 < time or int(time) != time:
            mother = t_df.loc[0,'마더코드']
            
            if mother in except_mother_list:
                pass
            else:
                raise ValueError
    
    # train에서 마더코드 100623, 40분으로 나온 부분 : 20분으로 수정
    t_index = df_[(df_['마더코드'] == 100623) & (df_['노출(분)'] == 40)].index
    df_.loc[t_index,'노출(분)'] = 20
    
    
    df_['w_노출(분)'] = np.round(df_['노출(분)'] / df_['origin_노출(분)'],3)
    
    # 소수점 안써도 되는 애들 체크
    # train[train['마더코드']==100623] : 마지막 방송을 2시 40분까지 함 -> 삭제?
    # 마더코드 100786 : 8월 19일에 20분 방송했는데 노출(분)이 30.1로 나옴 => 수정
    # 40을 제외한 나머지 애들은 15, 20, 30으로 만듦
    # 하다가 중간부터 안 팔린경우는 삭제해야하나?
    
    # test의 경우 30.1만 수정하면 됨
    index_list = df_[df_['노출(분)']==30.1].index
    df_.loc[index_list,'노출(분)']=30
        
    modify_time_list = np.unique(df_[df_['w_노출(분)']>1]['노출(분)']).tolist()[:-1]
    
    #ind_list = train[train['노출(분)'].isin(modify_time_list)].index
    
    #a = train[train['노출(분)']==modify_time_list[1]]
    #np.unique(a['w_노출(분)'])
    
    
    # 0번째 : np.round로 가능 -> 1.001
    # 1번째 : np.round로 가능 -> 1.003
    # 2번째 : np.round로 가능한 부분은 1.003, 1.505는 20으로 수정
    
    target_time_list = [1.001,1.003, 1.003]
    
    for i in range(len(modify_time_list)):
        target_time = target_time_list[i]
        modify_time = modify_time_list[i]
        
        t_df = df_[df_['노출(분)']==modify_time]
        index_list = t_df[t_df['w_노출(분)']==target_time].index
        
        df_.loc[index_list,'노출(분)'] = np.round(df_.loc[index_list,'노출(분)'],0)
        
        if i ==2:
            index_list = t_df[t_df['w_노출(분)']==1.505].index
            df_.loc[index_list,'노출(분)'] = 20
        
    
    
    df_['w_노출(분)'] = np.round(df_['노출(분)'] / df_['origin_노출(분)'],3)
    df_ = df_.drop('origin_노출(분)',axis=1)
    
    # 기준점 추가
    df_['w0.5_노출(분)'] = df_['w_노출(분)'].apply(lambda x: 1 if x<0.5 else 0)
    return df_


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


def make_str_min(x, interval=5):
    
    minute = x.minute
    minute_slot = 0
    quotient = int(60/interval)
    
    for i in range(quotient):
        if minute < (i+1)*interval:
            minute_slot = i
            break

    return '%sminute%s'%(interval, minute_slot)

def make_hour_with_min(df_, date_col):
    df_['min10'] = df_[date_col].apply(lambda x: make_str_min(x,10))
    df_['s_hour_min10'] = ['%s_%s'%(df_.loc[i,'s_hour'], df_.loc[i,'min10']) for i in range(len(df_))]
    df_['s_hour_min10'] = df_['s_hour_min10'].apply(lambda x:'%s:%s'%(x.split('_')[0][1:],  str(int(x.split('_')[1][8:])*10).zfill(2)))
    return df_

def processing_view(df_, view_file_name=''):
    # 시청률 데이터
        # 전처리 1 : 2:00 ~ 6:20 제외
        # 전처리 2 : 토요일 18:00~18:20 제외
    if view_file_name == '':
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
    del view_data
    
    view_data = pd.read_csv(processed_path+'view_data.csv', parse_dates=['date'])
    view_data['date'] = view_data['date'].apply(lambda x: x - datetime.timedelta(days=1) if x.hour <2 else x)
    
    view_data['week'] = view_data['date'].apply(lambda x: utils.make_week(x))
    view_data['s_hour'] = view_data['date'].apply(lambda x: utils.make_s_hour(x))
    view_data['just_holiday_weekend'] = view_data['date'].apply(lambda x: make_str_holiday_weekend(x))
    view_data['just_week'] = view_data['date'].apply(lambda x: make_str_just_week(x))
    view_data = make_hour_with_min(view_data, 'date')
    
    #df_ = view_data['date'].apply(lambda x: make_str_holiday_weekend(x))

    
    df_['new_방송일시'] = df_['new_방송일시'].astype('datetime64[ns]')
    df_['s_hour'] = df_['new_방송일시'].apply(lambda x: utils.make_s_hour(x))
    df_['just_holiday_weekend'] = df_['new_방송일시'].apply(lambda x: make_str_holiday_weekend(x))
    df_['just_week'] = df_['new_방송일시'].apply(lambda x: make_str_just_week(x))
    df_ = make_hour_with_min(df_, 'new_방송일시')
    
    
    m_view_hour_holiday_weekend = view_data.groupby(['s_hour','just_holiday_weekend']).mean().reset_index()
    m_view_hour_holiday_weekend.columns = ['s_hour','just_holiday_weekend','m_view_hour_holiday_weekend']
    
    m_view_hour_weekend = view_data.groupby(['s_hour','just_week']).mean().reset_index()
    m_view_hour_weekend.columns = ['s_hour', 'just_week', 'm_view_hour_weekend']
    
    m_view_hour_min10_holiday_weekend = view_data.groupby(['s_hour_min10','just_holiday_weekend']).mean().reset_index()
    m_view_hour_min10_holiday_weekend.columns = ['s_hour_min10','just_holiday_weekend','m_view_hour_min10_holiday_weekend']
    
    m_view_hour_min10_weekend = view_data.groupby(['s_hour_min10','just_week']).mean().reset_index()
    m_view_hour_min10_weekend.columns = ['s_hour_min10', 'just_week', 'm_view_hour_min10_weekend']
    
    df_ = pd.merge(df_, m_view_hour_holiday_weekend, on=['s_hour','just_holiday_weekend'], how='left')
    df_ = pd.merge(df_, m_view_hour_weekend, on=['s_hour', 'just_week'], how='left')
    df_ = pd.merge(df_, m_view_hour_min10_holiday_weekend, on=['s_hour_min10','just_holiday_weekend'], how='left')
    df_ = pd.merge(df_, m_view_hour_min10_weekend, on=['s_hour_min10','just_week'], how='left')
    
    df_['방송일시'] =df_['방송일시'].astype(str)
    df_ = df_.drop(['s_hour', 'just_holiday_weekend','just_week',
                    's_hour_min10','min10'],axis=1)
    
    return df_


# period, product_period 변수 생성
def processing_Period_ProductPeriod(df_):
    
    # period 변수: 하나의 방송안에서 구간을 나타내는 변수 (0을 포함한 자연수.) (0,1,2, ... )
    # product_period 변수: 동일한 상품이 여러번 등장할 경우 횟수를 나타내는 변수. (0을 포함한 자연수) (0: 첫번째 방송, 1: 두번째 방송, 2: 세번째 방송, ... ...)
    
    

    #1. 상품명으로 sorting
    df_ = df_.sort_values(by=['상품명','방송일시']).reset_index(drop=True)

    df_['방송일시']=pd.to_datetime(df_['방송일시']) # 방송일시 변수 데이터 타입 datetime으로 변경

    df_['period'] = 0 # 방송당 구간 번호를 나타내는 변수
    df_['product_period'] = 0 # 상품 재등장시 방송 번호 
    
    for i in range(1,len(df_)):

        if df_.loc[i,'상품명'] == df_.loc[i-1,'상품명']: #  상품명이 같다.
            if abs((df_.loc[i,'방송일시'] -df_.loc[i-1,'방송일시']).total_seconds()/60) < 41: # 방송날도 같다
                df_.loc[i,'period'] = df_.loc[i-1,'period'] + 1
                df_.loc[i,'product_period'] =  df_.loc[i-1,'product_period']
            else:
                df_.loc[i,'product_period'] =  df_.loc[i-1,'product_period'] +1
    print('BDS: "Look at the bright sight."')
    return df_


def processing_period_ratio(df_):
    
    # period_ratio 변수: 방송내 구간별 가중치 변수로써 0~1 값을 갖고, 구간을 n등분하여 가중치를 순차적으로 누적하여 매김.

    cumsum_period = 0
    cumsum_period_list = []

    for i in range(len(df_)):
        cur_period = df_.loc[i,'period']
        if cur_period ==0 and i !=0:
            cumsum_period_list.append(cumsum_period)
            cumsum_period=1

        else:
            cumsum_period+=1

        if i == len(df_)-1:
            cumsum_period_list.append(cumsum_period)



    temp_list = []
    for i in cumsum_period_list:
        for j in range(i):
            temp_list.append((j+1)/i)

    df_['period_ratio'] = temp_list
    
    
    return df_


def processing_w_period_ratio(df_):

    # w_period_ratio 변수: period_ratio의 단점을 보완한 변수로써 중간 값을 사용하여 기군 주간을 중앙으로 함. 0~1 사이의 값을 가짐.
    
    ratio_list = []


    past_ratio = 0

    for i in range(len(df_)):

        cur_ratio = df_.loc[i,'period_ratio']

        half_value = (cur_ratio-past_ratio)/2
        diff_value = cur_ratio-half_value
        ratio_list.append(diff_value)

        if cur_ratio == 1:  
            past_ratio = 0
            continue
        past_ratio = cur_ratio

    df_['w_period_ratio'] = ratio_list
    
    
    return df_



def processing_Monday_to_Sunday(df_):
    # 요일변수: Monday = 0 부터 Sunday = 6 까지의 값을 가짐.
    
    day_of_week = df_['방송일시'].dt.dayofweek
    df_['요일'] = day_of_week
    
    return df_


def processing_hour(df_):
    
    # 방송 시간대를 나타내는 변수
    df_['hour'] = df_['방송일시'].apply(lambda x: x.hour)
    
    return df_




def processing_weather(df_, file_name):
    
    weather = pd.read_csv(raw_path+file_name, encoding='euc-kr')
    weather = weather.fillna(0)
    # 전국평균집계
    tmp1=pd.DataFrame(weather['기온(°C)'].groupby(by=[weather['일시']]).mean())
    tmp2=pd.DataFrame(weather['강수량(mm)'].groupby(by=[weather['일시']]).mean())
    tmp3=pd.DataFrame(weather['습도(%)'].groupby(by=[weather['일시']]).mean())
    tmp4=pd.DataFrame(weather['현지기압(hPa)'].groupby(by=[weather['일시']]).mean())

    tmp1['일시'] = pd.to_datetime([i for i in tmp1.index])
    tmp2['일시'] = pd.to_datetime([i for i in tmp2.index])
    tmp3['일시'] = pd.to_datetime([i for i in tmp3.index])
    tmp4['일시'] = pd.to_datetime([i for i in tmp4.index])

    tmp1.reset_index(drop=True, inplace=True)
    tmp2.reset_index(drop=True, inplace=True)
    tmp3.reset_index(drop=True, inplace=True)
    tmp4.reset_index(drop=True, inplace=True)

    tmp1.columns = ['기온','일시']
    tmp2.columns = ['강수량','일시']
    tmp3.columns = ['습도','일시']
    tmp4.columns = ['기압','일시']

    # 변수 병합
    weather_df = pd.concat([tmp1, tmp2['강수량'], tmp3['습도'], tmp4['기압']], axis=1)

    weather_df['일시']=[str(i)[:13] for i in weather_df['일시']]
    df_['일시']=[str(i)[:13] for i in df_['방송일시']]

    df_= pd.merge(left=df_, right=weather_df, how='left', on='일시')

    return df_
	


def processing_new_date(df_):
    
    # 하루 편성을 06시 부터 02시라고 기준을 잡음. 
    # 00시, 01시, 02시인 경우 day를 -1 하여 기준을 맞춰줌.
    df_['new_방송일시'] = df_['방송일시']
    df_['new_요일'] = df_['요일']
    
    for i in df_.index:
        if df_.loc[i,'hour'] in [0,1,2]:
            df_.loc[i,'new_방송일시'] = df_.loc[i,'new_방송일시'] - datetime.timedelta(days=1)
            df_.loc[i,'new_요일'] = df_.loc[i,'new_방송일시'].weekday()
            
    return df_


def processing_new_month_day(df_):
    
    # new_방송일시를 기준으로 new_월, new_일 변수 생성
	df_['new_월']=0
	df_['new_일']=0

	for i in df_.index:
		df_.loc[i,'new_월'] = df_.loc[i,'new_방송일시'].month
		df_.loc[i,'new_일'] = df_.loc[i,'new_방송일시'].day
	
	return df_



def processing_week(df_):

	# 주차 변수 생성
	tmp = df_[['new_월','new_일','new_요일']].sort_values(by=['new_월','new_일','new_요일']).drop_duplicates(subset=['new_월','new_일','new_요일']).reset_index(drop=True)

	tmp['new_주차'] = 0
	idx = 0
	for i in tmp.index:
		if tmp.loc[i,'new_요일'] == 1: # 2019년 1월1일 화요일부터 시작하므로 화요일을 기준으로 7일씩 묶음.
			idx+=1
			tmp.loc[i,'new_주차'] = idx
		else:
			tmp.loc[i,'new_주차'] = idx
    
	df_ = pd.merge(df_, tmp, on=['new_월','new_일','new_요일'], how='inner')
	return df_


def processing_weekend_holiday(df_, date_col):
	holiday_col = 'holiday'
	weekend_col = 'weekend'
	if 'new' in date_col:
		holiday_col = 'new_holiday'
		weekend_col = 'new_weekend'
	df_[holiday_col] = df_[date_col].apply(lambda x: make_holiday(x))
	df_[weekend_col] = df_[date_col].apply(lambda x: make_weekend(x))
	return df_



def extract_holiday(year_, ymd_type=False):
    if year_ == 2019:
        holiday_list = ['20190101','20190204','20190205','20190206','20190301','20190506','20190606',
                    '20190815','20190912','20190913','20191003','20191009','20191225','20200101','20200606']
    if ymd_type == False:
        holiday_list = pd.to_datetime(holiday_list, format='%Y%m%d').tolist()
    return holiday_list

holiday_list = extract_holiday(2019, ymd_type=True)


def make_weekend(x):
    week = x.weekday()
       
    if week <5:
        return '0'
    
    else:
        
        return '1'

def make_holiday(x):

	week = x.weekday()
	
	year= str(x.year).zfill(4)
	month = str(x.month).zfill(2)
	day = str(x.day).zfill(2)
	ymd = year+month+day
	
	if ymd in holiday_list:
		return '1'
	else:
		return '0'


def make_season(x):
    month = x.month
    season = 0
    
    if month >=3 and month<6:
        season = 0
    elif month >=6 and month <9:
        season = 1
    elif month >=9 and month < 12:
        season = 2
    else:
        season = 3
    
    return season


def processing_team_idx(df_):
    
    # 방송마다 고유의 idx를 생성.
    # train/val set split 시에 방송별로 sampling하는데 사용한다.
    
    idx = 0
    df_['team_idx'] = 0
    for i in range(1, len(df_)):
        if (df_.loc[i,'상품명'] == df_.loc[i-1,'상품명']) and (df_.loc[i,'product_period'] == df_.loc[i-1,'product_period']):
            df_.loc[i,'team_idx'] = idx
        else:
            idx+=1
            df_.loc[i,'team_idx'] = idx

    return df_




def processing_season(df_, date_col_):
    season_col = 'season'
    if 'new' in date_col_:
        season_col = 'new_season'

    
    df_[season_col] = df_[date_col_].apply(lambda x: make_season(x))

    return df_




def make_bangsong_set(df_):
    
    # 방송 set 변수 생성
    df_ = df_.sort_values(by='방송일시') # 방송일시 정렬
    df_['방송일시'] = pd.to_datetime(df_['방송일시']) # 방송일시 datetime 변수 변경

    df_= df_.reset_index(drop=True)
    idx = 0
    df_['방송set'] = 0
    
    length = len(df_)
    for i in range(1,length):
        if (str(df_.loc[i,'마더코드'])[:-1] == str(df_.loc[i-1,'마더코드'])[:-1]) and ((abs((df_.loc[i,'방송일시'] - df_.loc[i-1,'방송일시'])).seconds//60) < 60) :
            df_.loc[i,'방송set']=idx
        else:
            idx+=1
            df_.loc[i,'방송set']=idx
    return df_



def processing_outlier(df_):
    
    df_['분당취급액'] = 0
    for i in df_.index:
        df_.loc[i,'분당취급액'] = df_.loc[i,'취급액']/df_.loc[i,'노출(분)']
        
    ###########방송set 변수 생성###########
    """
    df_2 = df_
    df_2 = df_2.sort_values(by='방송일시') # 방송일시 정렬
    df_2['방송일시'] = pd.to_datetime(df_2['방송일시']) # 방송일시 datetime 변수 변경

    df_2= df_2.reset_index(drop=True)
    idx = 0
    df_2['방송set'] = 0
    
    length = len(df_2)
    for i in range(1,length):
        if (df_2.loc[i,'마더코드'] == df_2.loc[i-1,'마더코드']) and ((abs((df_2.loc[i,'방송일시'] - df_2.loc[i-1,'방송일시'])).seconds//60) < 60) :
            df_2.loc[i,'방송set']=idx
        else:
            idx+=1
            df_2.loc[i,'방송set']=idx
    
    df_ = df_2
	"""
    df_ = make_bangsong_set(df_)
    
    df_2 = df_
    # 방송set 별로 분당취급액 합 데이터프레임 생성.
    tmp = pd.DataFrame(df_2['분당취급액'].groupby(by=[df_2['방송set']]).sum())
    tmp['방송set'] = [i for i in tmp.index]
    tmp.reset_index(drop=True, inplace=True)
    tmp

    # 원본데이터와 방송set 데이터 병합
    df_2= df_2.drop_duplicates(subset=['방송set'], keep='first').drop(columns=['분당취급액']).reset_index(drop=True)

    df_per_b = pd.merge(df_2, tmp, on=['방송set'])
    
    
	##########################################
	
    # 이상치: 상품군 별로 상하위 IQR * 1.5 보다 큰/작은 경우 또는 방송 set 취급액이 0인 상품.
    df_per_b['outlier_b'] = 0
    
    
    # 방송set 데이터프레임을 사용하여 boundary 생성.
    c_boundary_d = {} # {상품군: Q3+IQR*1.5} :상품군별 상위이상치 boundary 사전 생성.
    for i in set(df_per_b['상품군']): # 상품군 별 평균 dictionary
        Q1 = np.percentile(df_per_b[df_per_b['상품군']==i]['분당취급액'],25) # 1사분위수
        Q3 = np.percentile(df_per_b[df_per_b['상품군']==i]['분당취급액'],75) # 3사분위수
        IQR = Q3 - Q1 
        upper_outlier_boundary = Q3 + (IQR * 1.5) # 해당 상품군의 Q3 + IQR*1.5 위의 이상치.
        low_outlier_boundary= Q1 - (IQR * 1.5) # 해당 상품군의 Q1 - IQR*1.5 아래의 이상치.
        c_boundary_d[i] = []
        c_boundary_d[i].append(upper_outlier_boundary) # 상위이상치 제거
        c_boundary_d[i].append(low_outlier_boundary) # 하위이상치 제거
    
    
    # 방송set 데이터프레임의 이상치 변수 입력.
    for i in df_per_b.index:
        if (df_per_b.loc[i,'분당취급액'] == 0) or (df_per_b.loc[i,'분당취급액'] < c_boundary_d[df_per_b.loc[i,'상품군']][1]): #취급액이 0 또는 하위이상치이면 outlier_b = 1
            df_per_b.loc[i,'outlier_b'] = 1
        elif df_per_b.loc[i,'분당취급액'] > c_boundary_d[df_per_b.loc[i,'상품군']][0]: # 해당 상품군의 outlier_boundary 보다 높으면 outlier_b = 2
            df_per_b.loc[i,'outlier_b'] = 2

    # 방송set 데이터 프레임과 원본 데이터 프레임의 병합.

    df_ = pd.merge(df_,df_per_b[['new_월','new_일','마더코드','방송set','outlier_b']], how='left', on=['new_월','new_일','마더코드','방송set'])
    #df_ = df_[df_['outlier_b']==0]
    #df_ = df_.drop(columns=['outlier_b'])
    return df_



# 방송순서 가중치 생성
def processing_product_order_w(df_):
    
    df_ = df_.sort_values(by=['team_idx','period'])
    money_df = pd.DataFrame(df_['분당취급액'].groupby(by=[df_['team_idx'], df_['period']]).sum())
    money_df['team_idx'] = [i[0] for i in money_df.index]
    money_df['period'] = [i[1] for i in money_df.index]
    money_df.reset_index(drop=True, inplace=True)


    warnings.filterwarnings(action='ignore')
    # min-max 정규화.
    max_money = max(money_df.분당취급액)
    min_money = min(money_df.분당취급액)
    for i in money_df.index:
        money_df.loc[i,'분당취급액'] = (money_df.loc[i,'분당취급액'] - min_money) / (max_money - min_money)

    money_df.sort_values(by=['team_idx','period'], ascending=True, inplace=True)   
    money_df.reset_index(drop=True, inplace=True)
    
    length = len(money_df.index)
    money_df['상품구간개수_분류'] = 0

    basket=[]
    for i in  sorted(set(money_df['team_idx'])):
        answer = max(money_df[money_df['team_idx']==i]['period'])+1
        if answer != len(money_df[money_df['team_idx']==i]['period']):
            basket.extend([answer]*(answer-1)) # 1개
        else:
            basket.extend([answer]*answer)

    df_['상품구간개수_분류'] = basket
    
    df_.reset_index(drop=True, inplace=True)
    
    
    
    
    idx_d={}
    for i in sorted(set(df_['상품구간개수_분류'])): 
        idx_d[i]=df_[df_['상품구간개수_분류']==i]['분당취급액'].groupby(by=[df_[df_['상품구간개수_분류']==i]['period']]).mean()
    
    result_d = {}
    for i in idx_d: 
        diff_ratio=[]

        for j in idx_d[i].index: # 구간별 gradient 개산

            if len(idx_d[i]) == 1:
                result_d[1] = {0:1}
                
            else:
                for k in range(0,len(idx_d[i])-1):
                    diff_ratio.append(idx_d[i][k+1]/idx_d[i][k])
            result_d[i]  = {a:b for a,b in zip(list(idx_d[i].index)[:-1], list(diff_ratio)[:-1]) }
    result_d[1] = {0:1}     

    
    
    idx_d={}
    for i in sorted(set(df_['상품구간개수_분류'])): 
        idx_d[i]=df_[df_['상품구간개수_분류']==i]['분당취급액'].groupby(by=[df_[df_['상품구간개수_분류']==i]['period']]).mean()
    
    
    basket={}
    basket[1] = {0:1}
    for i in result_d:

        if i > 1:

            basket[i]={0:1/i}
            dum = 1/i
            for num in result_d[i].keys():
                dum *= result_d[i][num]
                basket[i][num+1] = dum


    for i in basket:
        if i > 1:

            s = sum(basket[i].values())
            for j in basket[i].keys():
                basket[i][j]=basket[i][j]/s



    idx_d = basket

    df_['상품구간_w'] = 0
    for i in df_.index:
        df_.loc[i,'상품구간_w'] = idx_d[df_.loc[i,'상품구간개수_분류']][df_.loc[i,'period']]

    df_.drop(columns=['상품구간개수_분류'])
    return df_

	



# 피크를 중심으로 3구간으로 분류하여 사인값을 적용한 시간대_w 변수
def processing_time3_w(df_):
    hour=list(range(0,6,1))
    hour_in_day = 6
    hour_sin  = np.sin([np.pi*i/hour_in_day for i in hour])
    hour_cos  = np.cos([np.pi*i/hour_in_day for i in hour])

    sin={}
    for i in [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0,1,2]:
        if i in [6, 1,2]:
            sin[i]=hour_sin[1:][0]
        if i in [7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,0]:
            sin[i]=hour_sin[1:][1]
        if i in [21,22]:
            sin[i]=hour_sin[1:][2]

    df_['시간대_w'] = 0
    df_.reset_index(drop=True, inplace=True)


    for i in df_.index:
        df_.loc[i,'시간대_w'] = sin[df_.loc[i,'hour']]

    return df_


# 피크 중심으로 사인함수 내리는 time_w 주기함수

def processing_time_w(df_):
    
    # 사인함수 구간 설정
    hour=list(range(0,11,1)) 
    hour_in_day = 11 
    
    hour_sin1  = np.sin([np.pi*i/hour_in_day for i in hour]).tolist()[10:2:-1] # 첫번째 피크
    hour_sin2  = np.sin([np.pi*i/hour_in_day for i in hour]).tolist()[8:3:-1] # 두번째 피크
    hour_sin3  = np.sin([np.pi*i/hour_in_day for i in hour]).tolist()[7::-1] # 세번째 피크
    
    hour_sin = hour_sin1 + hour_sin2 + hour_sin3 
    
    # 시간대를 주기값으로 변경해주는 사전 생성.
    sin={}
    for i in zip([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0,1,2], hour_sin):
        sin[i[0]]=i[1]
    
    df_['시간_w'] = 0
    df_.reset_index(drop=True, inplace=True)
    for i in df_.index:
        df_.loc[i,'시간_w'] = sin[df_.loc[i,'hour']]
        
    return df_
	
# 프라임시간대 21,22시 prime_time_w 가중치 변수

def processing_prime_time_w(df_):
    
    # 사인함수 구간 설정
    hour=list(range(0,5,1)) 
    hour_in_day = 5
    
    hour_sin = np.sin([np.pi*i/hour_in_day for i in hour]).tolist()
    
    # 시간대를 주기값으로 변경해주는 사전 생성.
    sin={}
    for i in [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0,1,2]:
        if i in [1,2]: # 새벽 구간 1,2
            sin[i]=hour_sin[0]
        elif i in [6]: # 기상 구간 6 
            sin[i]=hour_sin[4]
        elif i in list(range(7,21,1)): # 활동 구간 7,8,9,10,11,12,13,14,15,16,17,18,19,20
            sin[i]=hour_sin[1]
        elif i in [23,0]: # 취침 구간 23, 00
            sin[i]=hour_sin[3]
        elif i in [21,22]: # 피크 구간 21,22
            sin[i]=hour_sin[2]
    
    df_['프라임시간_w'] = 0
    df_.reset_index(drop=True, inplace=True)

    for i in df_.index:
        df_.loc[i,'프라임시간_w'] = sin[df_.loc[i,'hour']]
        
    return df_


def processing_time(df_):
    hour=list(range(0,21,1)) 
    hour_in_day = 21
    hour_sin = np.sin([np.pi*i/hour_in_day for i in hour]).tolist()[::-1]

    sin_d = {}
    for a,b in zip([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0,1,2],hour_sin):
        sin_d[a] = b

    df_['시간주기']=0
    for i in df_.index:
        df_.loc[i,'시간주기'] = sin_d[df_.loc[i,'hour']]
        
    return df_


def change_column_name(df_):
    
    delete_list = ['same','상품코드','요일','일시','season','holiday','weekend',
                   'ymd','방송순서_비율','brand_w']
    
    for col in delete_list:
        if col in df_.columns:
            df_ = df_.drop(col, axis=1)
    
    col_dict = {'period':'방송상품idx', 'product_period':'상품별_idx',
                'period_ratio':'방송순서_비율','w_period_ratio':'방송순서_w',
                'team_idx':'방송상품set_idx', 'hour':'시간대',
                '기온(°C)':'기온','강수량(mm)':'강수량','풍속(m/s)':'풍속',
                '습도(%)':'습도', 'w_노출(분)':'노출(분)_w',
                'new_상품명':'상품명_n',
                'm_view_hour_holiday_weekend':'주중휴일별_시간별_시청률',
                'm_view_hour_weekend':'주중별_시간별_시청률',
                'm_view_hour_min10_holiday_weekend':'주중휴일별_시간별_10분별_시청률',
                'm_view_hour_min10_weekend':'주중별_시간별_10분별_시청률',
                'new_월':'월','new_일':'일','new_주차':'주차','new_요일':'요일',
                'new_season':'계절','new_방송일시':'방송일시n',
                'new_holiday':'휴일','new_weekend':'주말'}
    
    columns = df_.columns.tolist()
    for i in range(len(columns)):
        col = columns[i]
        if col in col_dict.keys():
            new_col = col_dict[col]
            columns[i] = new_col
        else:
            continue
    
    df_.columns = columns
    return df_


# Get traffic data
def get_traff(path, trans):
    all_files = glob.glob(path + '\\{}\\*.csv'.format(trans))
    file_li = []
    
    for i in all_files:
        try:
            if '202006' not in i:
                file_li.append(pd.read_csv(i, 
                                           index_col=None,
                                           header=0,
                                           encoding='CP949'))
            else:    
                temp_df = pd.read_csv(i,index_col=None,
                                      header=0,
                                      encoding='CP949')
                temp_df.columns = ['노선ID','노선번호','노선명',
                                   '버스정류장ARS번호','역명',
                                   '승차총승객수','하차총승객수',
                                   '등록일자']
                temp_df['사용일자'] = temp_df.index
                temp_df =temp_df.reset_index(drop=True)
                file_li.append(temp_df)
                del temp_df
        except(UnicodeError):
            if '202006' not in i:
                file_li.append(pd.read_csv(i,
                                           index_col=None,
                                           header=0,
                                           encoding='utf8'))
            else:
                temp_df = pd.read_csv(i,index_col=None,
                                      header=0,
                                      encoding='utf8')
                temp_df.columns=['사용일자','노선명','역명',
                                 '승차총승객수','하차총승객수',
                                 '등록일자','none']
                temp_df = temp_df.drop('등록일자',axis=1)
                file_li.append(temp_df)
                del temp_df
                
                
    df = pd.concat(file_li, ignore_index=True)
    
    df_sum = df.groupby('사용일자').sum()
    df_sum.reset_index(inplace=True)
    
    peo = []
    in_peo = df_sum['승차총승객수'].tolist()
    out_peo = df_sum['하차총승객수'].tolist()
    
    for i, j in zip(in_peo, out_peo):
        peo.append(np.round((i + j) / 2, 2))
        
    df_sum['대중교통_이용_승객수'] = peo
    
    df_sum['사용일자'] = df_sum['사용일자'].apply(lambda x: str(x))
    df_sum['사용일자'] = df_sum['사용일자'].apply(lambda x: datetime.datetime(int(x[:4]),
                                                                      int(x[4:6]),
                                                                      int(x[6:])))
    result = df_sum[['사용일자','대중교통_이용_승객수']]
    
    return result


# Make traffic with sales
def make_traff_df(df_, all_traff):

    df_['ymd'] = df_['방송일시'].apply(lambda x: utils.make_ymd(x))
    all_traff['ymd'] = all_traff['사용일자'].apply(lambda x: utils.make_ymd(x))
    all_traff = all_traff.drop('사용일자', axis=1)
    df_ = pd.merge(df_, all_traff, on='ymd', how='left')

    return df_


def processing_make_transport(df_):
    if os.path.isfile(processed_path+'all_traff.csv') == False:
        sub_traff = get_traff(raw_path, 'subway')
        bus_traff = get_traff(raw_path, 'bus')
        
        all_traff = [i + j for i, j in zip(sub_traff['대중교통_이용_승객수'], 
                                           bus_traff['대중교통_이용_승객수'])]
        
        all_traff = pd.DataFrame({'사용일자':bus_traff['사용일자'],
                                 '대중교통':all_traff})
        
        all_traff = all_traff.append({'사용일자':datetime.datetime(2020,1,1),
                                      '대중교통':all_traff.iloc[-1,1]},
                                     ignore_index=True)
        all_traff = all_traff.sort_values('사용일자').reset_index(drop=True)
        all_traff.to_csv(processed_path+'all_traff.csv', index=False)
        
    else:
        all_traff = pd.read_csv(processed_path+'all_traff.csv',
                                parse_dates=['사용일자'])
        
    df_ = make_traff_df(df_, all_traff)
    
    return df_



def processing_air(df_):
    try:
        air_df = pd.read_csv(processed_path+'air_train_test.csv', parse_dates=['time'])
    except(FileNotFoundError):
        make_air_pollution_data()
        air_df = pd.read_csv(processed_path+'air_train_test.csv', parse_dates=['time'])
        
    air_df['ymd'] = air_df['time'].apply(lambda x: utils.make_ymd(x))
    air_df = air_df.drop('time',axis=1)
    df_ = pd.merge(df_, air_df, on='ymd', how='left')
    
    return df_


def brand_weight(df_):
    
    
    brand = pd.read_csv(raw_path+'브랜드명.csv').iloc[:,1]
    
    # brand 변수: 상품명에 브랜드 명이 포함되어 있는 경우, brand = 1 그렇지 않으면 brand = 0 인 이진 변수.
    # brand_w 변수: 브랜드 상품이 해당 상품군 내에서 평균보다 높은 취급액을 가질 경우의 해당 브랜드의 가중치를 주는 변수.
    
    df_['분당취급액'] = 0
    for i in df_.index:
        df_.loc[i,'분당취급액'] = df_.loc[i,'취급액']/df_.loc[i,'노출(분)']
	
    # 하나의 방송당 취급액 합으로써 방송시간에 하나의 고유한 row만 갖게 됨.
    tmp = pd.DataFrame(df_['분당취급액'].groupby(by=[df_['team_idx']]).sum())
    tmp2 = df_.drop_duplicates(subset=['상품명','team_idx'], keep='first').drop(columns=['분당취급액']).reset_index(drop=True)
    df_per_b = pd.concat([tmp2, tmp], axis=1)
    anal = df_per_b
    brand_weight_dict = {}
    tmp = anal.drop_duplicates(subset=['상품명']).reset_index(drop=True)
    # 상품군별 브랜드 사전 생성. 가중치x
    for i in tmp.index:
        for brand_ in brand:
            if brand_ in tmp.loc[i,'상품명']:
                if tmp.loc[i,'상품군'] not in brand_weight_dict.keys():
                    brand_weight_dict[tmp.loc[i,'상품군']] = {brand_: 0}
                else:
                    brand_weight_dict[tmp.loc[i,'상품군']][brand_]= 0

    # 상품군에 대해 불필요하게 겹치는 브랜드명 제거 (수작업)
    
    if '캐리어' in brand_weight_dict['속옷'].keys():
        # : 상품군 속옷에 '캐리어'는 브랜드가 아니므로 제거.
        del brand_weight_dict['속옷']['캐리어']
    
    if '크로커다일' in brand_weight_dict['주방'].keys():
        del brand_weight_dict['주방']['크로커다일']

    x=pd.DataFrame(brand_weight_dict)

    for i in x.columns:
        brand_weight_dict[i]={}
        for j in x[x[i]==0][[i]].index:
            brand_weight_dict[i][j]=0


    anal['brand'] = 0


    for i in anal.index:
        brand_in_class = brand_weight_dict[anal.loc[i,'상품군']].keys()
        for j in brand_in_class:
            if j in anal.loc[i,'상품명']: # 상품명에 브랜드가 포함되어 있으면, brand = 1
                anal.loc[i,'brand'] = 1
                break


    brand_scale = {} # brand 가중치 조절 파라미터. 브랜드 포함 비율이 높을수록 brand_w 의 영향력을 크게 해주기 위함.

    for i in brand_weight_dict:
        num_brand = len(anal.loc[(anal['상품군']==i) & (anal['brand']==1)])
        brand_scale[i] = num_brand/len(anal[anal['상품군']==i])

    ### 상품군 별로 브랜드 포함 비율을 확인하고, 이를 brand_w 영향력 기준.
    #-> 브랜드 포함 비율이 많다는 것은 그만큼 브랜드 경쟁이 치열하다고 볼 수 있으며, 브랜드 파워의 영향력은 해당 상품군에서 더 클 것으로 봄.

    
    for i in brand_weight_dict:
        tmp = anal.loc[(anal['상품군']==i)] 
        class_avg = np.mean(tmp['분당취급액']) # 상품군의 분당취급액 평균값.

        for j in brand_weight_dict[i]: # 상품군 내에 브랜드를 탐색
            if class_avg < np.mean(tmp[tmp['상품명'].apply(lambda x: j in x)]['분당취급액']):   # 평균 보다 크면 브랜드 가중치를 주게됨.
                brand_weight_dict[i][j] = (np.mean(tmp[tmp['상품명'].apply(lambda x: j in x)]['분당취급액'])) / class_avg * brand_scale[i] # 상품군별 브랜드 영향력에 맞추어 가중치를 주게됨.
            else:
                brand_weight_dict[i][j] = 0

    ### brand_w 변수를 생성.

    anal['brand_w'] = 0
    for i in anal.index:
        for j in brand_weight_dict[anal.loc[i,'상품군']].keys():
            if j in anal.loc[i,'상품명']:
                anal.loc[i,'brand_w'] = brand_weight_dict[anal.loc[i,'상품군']][j]
                break
    
    df_['brand_w'] = 0
    for i in df_.index:
        for j in brand_weight_dict[df_.loc[i,'상품군']].keys():
            if j in df_.loc[i,'상품명']:
                df_.loc[i,'brand_w'] = brand_weight_dict[df_.loc[i,'상품군']][j]
                break
    
    return df_



#%%


train_path = utils.train_path
processed_path = utils.processed_path
raw_path = utils.raw_path

def main():
    start = time.time()
    file_type_list = ['train','test']
    file_name_list = ['2020 빅콘테스트 데이터분석분야-챔피언리그_2019년 실적데이터_v1_200818.xlsx',
                 '2020 빅콘테스트 데이터분석분야-챔피언리그_2020년 6월 판매실적예측데이터(평가데이터).xlsx']
    save_name_list = ['train.csv','test.csv']
    weather_name_list = ['날씨데이터.csv','날씨데이터.csv']
    
    # file 검사
    train_folder_list = os.listdir(train_path)
    count=0
    
    for save_name in save_name_list:
        if save_name in train_folder_list:
            count+=1
    
    if count >= 1:
        while True:
            str_ = input('You have at least one file that is same name now. continue?')
            if str_.lower() == 'n':
                return 0
            elif str_.lower() == 'y':
                print('ok')
                break
            else:
                print('please input y or n')

    for i in range(2):
        file_type = file_type_list[i]
        file_name = file_name_list[i]
        save_name = save_name_list[i]
        weather_name = weather_name_list[i]
    
        df = pd.read_excel(raw_path+file_name, header=1)
        df['노출(분)'] = df['노출(분)'].fillna(method='ffill')
        df['방송일시'] = df['방송일시'].astype('datetime64[ns]')
        df = df[df['상품군']!='무형'].reset_index(drop=True)
        if file_type == 'train':
            df['취급액'] = df['취급액'].fillna(0)
        
        
        df = processing_nocul_time(df, file_type)
        
        # period, product_period 변수 생성
        df = processing_Period_ProductPeriod(df)
        
        # period_ratio 변수 생성
        df = processing_period_ratio(df)
        
        # w_period_ratio
        df = processing_w_period_ratio(df)
        
        # 요일 변수 추가
        df = processing_Monday_to_Sunday(df)
        
        # 방송 시간 변수 추가 
        df = processing_hour(df)
        
        # 날씨변수 생성(기온, 습도, 풍속, 강수량)
        df = processing_weather(df, weather_name)
        
        # new 방송일시 생성
        df = processing_new_date(df)
        
        # 계절변수 생성(기존 date)
        df = processing_season(df, '방송일시')
        # 계절변수 생성(new_date)
        df = processing_season(df, 'new_방송일시')
		
		# 휴일변수 생성(기존 date)
        df = processing_weekend_holiday(df, '방송일시')
        # 휴일변수 생성(new_date)
        df = processing_weekend_holiday(df, 'new_방송일시')
		
		# new_방송일시를 기준으로 new_월, new_일 변수 생성
        df = processing_new_month_day(df)
		
		# 주차 변수 생성
        df = processing_week(df)
		
		# team_idx 생성
        df = processing_team_idx(df)
		
		# wor2_vec key값 생성
        df['word_key'] = df.index
        
        df = brand_weight(df)
        
        # order_w
        df = processing_product_order_w(df)

        # 방송변수 생성
        df = processing_view(df)
        
        # 시간주기
        df = processing_time(df)
        
        # 시간_w
        df = processing_prime_time_w(df)
		
        # 프라임시간_w
        df = processing_time_w(df)
		
        # 시간대_w
        df = processing_time3_w(df)

        
        if file_type == 'train':
            df = processing_outlier(df)
            
            # column 이름 바꾸기
            df = change_column_name(df)

        else:
            # test 방송set 만들기
            df = make_bangsong_set(df)
            # column 이름 바꾸기
            df = change_column_name(df)
        
        
        # 대중교통 변수 생성
        df = processing_make_transport(df)
        
        # 날씨변수 생성
        df = processing_air(df)
		
        # ymd 제거
        df = df.drop('ymd',axis=1)
        
        # outlier 포함된 파일 먼저 저장
        if file_type == 'train':
            df.to_csv(train_path+'train_with_outlier.csv')
            df = df[df['outlier_b']==0].reset_index(drop=True)
            df = df.drop(columns=['outlier_b'])
        
        
        # word2_vec 변수 넣기
        df['word_key'] = [i for i in range(len(df))]
        
        
		# 저장
        df.to_csv(train_path+save_name, index=False)
        print('%s완료'%(file_type))
        print("최종 time :", time.time() - start)
    return 1

if __name__ =='__main__':
    main()
    


