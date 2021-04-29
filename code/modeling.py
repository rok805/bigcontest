# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:48:01 2020

@author: User
"""


import pandas as pd
import numpy as np
import os
import utils
from sklearn.linear_model import SGDRegressor
from load_data import load_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import configparser
from model import randomforest, mlp, xgboosting
import sys
import pdb
import shutil
import pickle
import torch


# 수요량 편집
raw_path = utils.raw_path
processed_path = utils.processed_path
train_path = utils.train_path


"""
dummy = ['마더코드', '상품코드','상품군','hour','요일','season','holiday'
         'new_월','new_일','new_요일','new_주차','new_season']
"""

def make_dummy(df, col_name):
    #col_name = 'new_월'
    dummy = pd.get_dummies(df[col_name], prefix=col_name)
    
    dummy_df = pd.DataFrame({'name':df[col_name]})
    dummy_df = pd.concat([dummy_df, dummy], axis=1)
    
    name_dict = {}
    for col in dummy_df.columns[1:]:
        target = dummy_df[dummy_df[col]!=0].reset_index(drop=True).loc[0,['name',col]]
        name_dict[col] = target['name']
    
    col_stand_name = '_'.join(dummy_df.columns[1].split('_')[:-1])
    col_list = ['%s_%s'%(col_stand_name,v) for v in name_dict.values()]
    
    dummy_df = dummy_df.iloc[:,1:]
    dummy_df.columns = col_list
    
    df = df.drop(col_name, axis=1)
    df = pd.concat([df, dummy_df], axis=1)
    return df


def train_test_split(x,y, train_ind, valid_ind):
    
    X_train,y_train,X_valid,y_valid = [],[],[],[]
    
    for i in range(len(train_ind)):
        t_train_ind, t_valid_ind = train_ind[i], valid_ind[i]
        t_X_train, t_y_train = x.loc[t_train_ind,:].values, y.loc[t_train_ind,:].values.reshape((-1,1))
        t_X_valid, t_y_valid = x.loc[t_valid_ind,:].values, y.loc[t_valid_ind,:].values.reshape((-1,1))

        X_train.append(t_X_train)
        y_train.append(t_y_train)
        X_valid.append(t_X_valid)
        y_valid.append(t_y_valid)

    return X_train, y_train, X_valid, y_valid

    
def split_index(day_list, num, random_state):
    
    train_list = np.random.choice(day_list, num, replace=False).tolist()
    
    day_list = np.setdiff1d(day_list, train_list).tolist()
    return train_list, day_list

def result_file_copy(save_path, config_file_name):
    shutil.copy('config/%s'%(config_file_name), '%s/model_setting.txt'%(save_path))
    shutil.copy('modeling.py', '%s/modeling.py'%save_path)
    shutil.copy('model.py', '%s/model.py'%save_path)




def sampling(origin_df, split_set_option, train_rate, valid_rate, random_state=0):
    
    df = origin_df.copy()
    # test 여부
    
        
    ptype_list = np.unique(df['상품군'])
        
    train_set_list = []
    valid_set_list = []
    month_list = np.unique(df['월'])
    
    np.random.seed(random_state) 
    
    if split_set_option == 'y':
    
        for month in month_list:
            month_df = df[df['월']==month].reset_index(drop=True)
            for ptype in ptype_list:
                temp_df = month_df[month_df['상품군']==ptype].reset_index(drop=True)
                
                set_list = np.unique(temp_df['방송set'])
                len_set = len(set_list)
                
                len_train = int(train_rate*len_set)
                len_valid = len_set - len_train
                            
                valid_set = np.random.choice(set_list, len_valid, replace=False).tolist()
                set_list = np.setdiff1d(set_list, valid_set).tolist()
                train_set = set_list.copy()
                            
                train_set_list += train_set
                valid_set_list += valid_set
        
        final_train_index = df[df['방송set'].isin(train_set_list)].index.tolist()
        final_valid_index = df[df['방송set'].isin(valid_set_list)].index.tolist()
        
    else:
        
        for month in month_list:
            month_df = df[df['월']==month]
            for ptype in ptype_list:
                temp_df = month_df[month_df['상품군']==ptype]
                index_list = temp_df.index.tolist()
                
                len_temp_df = len(temp_df)
                
                len_train = int(train_rate*len_temp_df)
                len_valid = len_set - len_train
                            
                valid_set = np.random.choice(index_list, len_valid, replace=False).tolist()
                index_list = np.setdiff1d(index_list, valid_set).tolist()
                train_set = index_list.copy()
                            
                train_set_list += train_set
                valid_set_list += valid_set
        
        final_train_index = train_set_list
        final_valid_index = valid_set_list

    return [final_train_index], [final_valid_index]



def cv_sampling(origin_df, split_set_option, cv_num, random_state=0):
    
    df = origin_df.copy()

    train_set_list = []
    valid_set_list = []   
    
    month_list = np.unique(df['월'])
    np.random.seed(random_state) 
    
    len_df = len(df)
    
    for cv in range(cv_num):
        train_set_list.append([])
        valid_set_list.append([])
    
    if split_set_option == 'y':
    
        for month in month_list:
            month_df = df[df['월']==month].reset_index(drop=True)
            ptype_list = np.unique(month_df['상품군'])
            for ptype in ptype_list:
                temp_df = month_df[month_df['상품군']==ptype].reset_index(drop=True)
                
                set_list = np.unique(temp_df['방송set'])
                temp_set_list = set_list.copy()
                len_set = len(set_list)
                
                len_valid = int(len_set/cv_num)
                
                for cv in range(cv_num):
    
                    if cv != cv_num -1:
                        valid_set = np.random.choice(temp_set_list, len_valid, replace=False).tolist()
                    else:
                        valid_set = temp_set_list
                        
                    train_set = np.setdiff1d(set_list, valid_set).tolist()
                    train_set_list[cv] +=train_set
                    valid_set_list[cv] +=valid_set
                    
                    temp_set_list = np.setdiff1d(temp_set_list, valid_set).tolist()
        
        
        final_train_index = []
        final_valid_index = []

        for cv in range(cv_num):
            final_train_index.append(df[df['방송set'].isin(train_set_list[cv])].index.tolist())
            final_valid_index.append(df[df['방송set'].isin(valid_set_list[cv])].index.tolist())

    
    else :
        
        for month in month_list:
            month_df = df[df['월']==month]
            ptype_list = np.unique(month_df['상품군'])
            for ptype in ptype_list:
                temp_df = month_df[month_df['상품군']==ptype]
                
                index_list = temp_df.index.tolist()
                
                temp_index_list = index_list.copy()
                len_temp_df = len(index_list)
                
                len_valid = int(len_temp_df/cv_num)
                
                for cv in range(cv_num):
                    
                    if cv != cv_num -1:
                        valid_set = np.random.choice(temp_index_list, len_valid, replace=False).tolist()
                    else:
                        valid_set = temp_index_list
                    
                    train_set = np.setdiff1d(index_list, valid_set).tolist()
                    train_set_list[cv] +=train_set
                    valid_set_list[cv] +=valid_set
                    
                    temp_index_list = np.setdiff1d(temp_index_list, valid_set).tolist()
        
        final_train_index = []
        final_valid_index = []
        
        for cv in range(cv_num):
            final_train_index.append(train_set_list[cv])
            final_valid_index.append(valid_set_list[cv])
    
    return final_train_index, final_valid_index
    
    
def main():

    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    config_file = open(sys.argv[1], encoding="euc-kr")
    config.readfp(config_file)
    """
    
    x_col = config['settings']['x_col']
    x_col = x_col.split(',')
    dummy = config['settings']['dummy']
    dummy = dummy.split(',')
    if dummy[0] == '':
        dummy = []
    y_col = config['settings']['y_col']
    y_col = y_col.split(',')
    word_option = config['settings']['word_option']
    save_folder = config['settings']['save_folder']
    save_name = config['settings']['save_name']
    normal_option = config['settings']['normal_option']
    rate = config['settings']['train_valid_test']
    train_rate, valid_rate = rate.split(',')
    train_rate, valid_rate = float(train_rate), float(valid_rate)
    model = config['settings']['model']
    random_state = int(config['settings']['random_state'])
    cv_value = int(config['settings']['cv_value'])
    train_file_name = config['settings']['train_file_name']
    split_set_option = config['settings']['split_set_option']
    config_file_name = config['settings']['config_file_name']
    
    save_path = utils.model_path+'%s/'%save_folder
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    
    result_file_copy(save_path, config_file_name)
    
    if word_option == 'y':
        df = load_data(train_file_name, 'train_WordVec.pkl')
        word_list = ['w2v%s'%i for i in range(300)]
        x_col += word_list
    
    elif word_option == 'n' :
        df = load_data('train.csv', word_option=False)
    
    df=df.fillna(0)
    
    #date_df = df[['new_방송일시']]
    #date_df.columns = ['date']
    
    x_data = df[x_col]
    y_data = df[y_col]
    
    for col in dummy:
        x_data = make_dummy(x_data, col)
    
    final_x_col = x_data.columns.tolist()
    x_data = x_data.loc[:,final_x_col]
    # 난수 지정    
    if cv_value >1:
        train_ind_list, valid_ind_list = cv_sampling(df,split_set_option,
                                                     cv_value,random_state)
    
    else:
        train_ind_list, valid_ind_list = sampling(df,split_set_option,
                                                  train_rate,valid_rate,
                                                  random_state)
    
    #pdb.set_trace()
    X_train, y_train, X_valid, y_valid = train_test_split(x_data,y_data,
                                                          train_ind_list,
                                                          valid_ind_list)
    
    #pdb.set_trace()
    scaler_list = []
    scaler=''
    if normal_option.lower() != 'none':
        for i in range(len(X_train)):
            if normal_option == 'standard':
                scaler = StandardScaler()
            elif normal_option == 'minmax':
                scaler = MinMaxScaler()
            
            scaler.fit(X_train[i])
            X_train[i] = scaler.transform(X_train[i])
            X_valid[i] = scaler.transform(X_valid[i])
            scaler_list.append(scaler)
            
    del x_data, y_data
    dataset = {'train':[X_train,y_train], 'valid':[X_valid, y_valid]}
    
    if model == 'rf':
        model_, result_dict = randomforest(dataset,config)
        feature_df = pd.DataFrame({'col':x_cols,'importance':np.round(model_.feature_importances_,4)})
        feature_df = feature_df.sort_values('importance', ascending=False).reset_index(drop=True)
        with open(save_path+'/%s.pkl'%(save_name), 'wb') as f:
            pickle.dump({'model':model_,'feature_df': feature_df}, f)
    
    elif model == 'mlp':
        result_dict = mlp(dataset,config, save_path)
    
    
    elif model == 'xgb':
        result_dict = xgboosting(dataset,config,save_path)

    else:
        raise ValueError
        
    result_dict['data'] = {}
    result_dict['data']['dataset'] = {'train':train_ind_list,'valid':valid_ind_list}
    result_dict['data']['x_col'] = x_col
    result_dict['data']['changed_x_col'] = final_x_col
    result_dict['data']['scaler'] = scaler_list
    with open(save_path+'/%s_result.pkl'%(save_name), 'wb') as f:
        pickle.dump({'result': result_dict}, f)
    
    
    mae = result_dict['score']['mae']
    mape = result_dict['score']['mape']

    print('mae : %.3f, mape : %.3f'%(mae,mape))
    
    return 0    
    

if __name__ == '__main__':
    main()
    
    
