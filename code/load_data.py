# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:16:20 2020

@author: User
"""

import pandas as pd
import pickle
import utils
import numpy as np

def load_data(data_name='', word_name='', word_option=True):
    """
    data_name : train이나 test의 csv 파일 이름(확장자 포함)
    word_name : train이나 test word 파일의 pkl 이름(확장자 포함)
    word_option : True면 word_vector 붙여서, False면 붙이지 않고 반환함
    """
    
    train_path = utils.train_path
    df = pd.read_csv(train_path+data_name)
    if word_option==True:
        with open(train_path+word_name, 'rb') as f:
            word_vector = pickle.load(f)
        
        word_col_list = ['w2v%s'%(i) for i in range(300)]
        
        ok_word_list = df['word_key'].tolist()
        word_vector = np.array(word_vector)
        word_vector = word_vector[ok_word_list,:]
        
        temp_df = pd.DataFrame({'word_key':ok_word_list})
        temp_df = pd.concat([temp_df, pd.DataFrame(word_vector)], axis=1)
        temp_df.columns = ['word_key']+word_col_list
        
        df = pd.merge(df, temp_df, on='word_key', how='left')
        
    return df
	

def colab_load_data(data_name='', word_name='', word_option=True):
    """
    data_name : train이나 test의 csv 파일 이름(확장자 포함)
    word_name : train이나 test word 파일의 pkl 이름(확장자 포함)
    word_option : True면 word_vector 붙여서, False면 붙이지 않고 반환함
    """
    root_path = '/content/drive/My Drive/bigcontest/workspace/'
    train_path = root_path+'data/train/'
    df = pd.read_csv(train_path+data_name)
    if word_option==True:
        with open(train_path+word_name, 'rb') as f:
            word_vector = pickle.load(f)
        
        word_col_list = ['w2v%s'%(i) for i in range(300)]
        
        ok_word_list = df['word_key'].tolist()
        word_vector = np.array(word_vector)
        word_vector = word_vector[ok_word_list,:]
        
        temp_df = pd.DataFrame({'word_key':ok_word_list})
        temp_df = pd.concat([temp_df, pd.DataFrame(word_vector)], axis=1)
        temp_df.columns = ['word_key']+word_col_list
        
        df = pd.merge(df, temp_df, on='word_key', how='left')
        
    return df

def df_per_b(data_name='', word_name='', word_option=True): # anal dataset: sumation of 'sold variable' each broadcast
    
    df = load_data(data_name, word_name, word_option)
    
    df['분당취급액'] = 0
    for i in df.index:
        df.loc[i,'분당취급액'] = df.loc[i,'취급액']/df.loc[i,'노출(분)']
        
    # 하나의 방송당 취급액 합으로써 방송시간에 하나의 고유한 row만 갖게 됨.
    tmp = pd.DataFrame(df['분당취급액'].groupby(by=[df['월'], df['일'],df['시간대'],df['상품군']]).sum())
    tmp['월'] = [i[0] for i in tmp.index]
    tmp['일'] = [i[1] for i in tmp.index]
    tmp['시간대'] = [i[2] for i in tmp.index]
    tmp['상품군'] = [i[3] for i in tmp.index]
    tmp.reset_index(drop=True, inplace=True)
    tmp2 =  df.drop_duplicates(subset=['월','일','시간대','상품군'], keep='first').sort_values(by='방송일시n').drop(columns=['분당취급액'])

    df_per_b = pd.merge(tmp2, tmp, on=['월','일','시간대','상품군'])
    return df_per_b


