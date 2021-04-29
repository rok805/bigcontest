"""
최적 홈쇼핑 방송 편성
"""

from calendar import monthrange
from collections import Counter
from load_data import load_data
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta, date
import re

# Personal modlues
from processing_train_test import processing_new_date
import processing_train_test as ptt
import utils

# Output path
output_path = utils.output_path
train_path = utils.train_path
processed_path = utils.processed_path
model_path = utils.model_path


# Fixed
def fixed_days():
    month = input('편성하고자 하는 년,월을 입력하세요. ex) 2020 7   :')

    # Make days by input
    year = int(month.split(' ')[0])
    mon = int(month.split(' ')[1])
    start = datetime(year, mon, 1, 6, 0)

    all_days = [start]
    while start.month != mon + 1:
        start = start + timedelta(hours=1)
        if start.hour == 3:
            start = start + timedelta(hours=3)
        if start.month != mon + 1:
            all_days.append(start)

    df = pd.DataFrame(all_days, columns=['방송일시'])
    df['hour'] = df['방송일시'].apply(lambda x: x.hour)
    df['요일'] = df['방송일시'].apply(lambda x: x.weekday())

    # Make new date
    df_ = processing_new_date(df)
    df_ = df_.drop(['방송일시', '요일'], axis=1)
    df_ = df_[['new_방송일시', 'hour', 'new_요일']]

    df_['주말'] = df_['new_방송일시'].apply(lambda x: ptt.make_weekend(x))

    holiday_list = input().split(', ')  # ex) 20191225, 20200101
    df_['휴일'] = df_['new_방송일시'].apply(lambda x: ptt.make_holiday(x))
    df_['계절'] = df_['new_방송일시'].apply(lambda x: ptt.make_season(x))

    # time_w
    df_ = ptt.processing_time_w(df_)

    # primetime_w
    df_ = ptt.processing_prime_time_w(df_)

    df_.columns = ['방송일시', '시간대', '요일', '주말', '휴일', '계절',
                   '시간_w', '프라임시간_w']

    return df_, mon


# Merge with 2019 data
def fixed_all(mon):

    # To get external
    # Load train data
    train_df = load_data('train.csv', '', False)

    # 휴일은 따로 지정해줘야함
    need_col = ['방송일시n', '노출(분)', '상품군', '판매단가',
                '기온', '강수량', '습도', '상품구간_w', '대중교통',
                '일시불_무이자', '일시불_할인율', 'PM10', '방송set', '상품명']
    train_df = train_df[need_col]
    train_df['방송일시'] = train_df['방송일시n'].apply(lambda x: datetime(int(x[:4]),
                                                                  int(x[5:7]),
                                                                  int(x[8:10]),
                                                                  int(x[11:13])))
    train_df.drop(['방송일시n'], axis=1, inplace=True)
    
    external = ['방송일시', '기온', '강수량', '습도', '대중교통', 'PM10']
    y_exter = train_df[external]
    
    m_exter = y_exter.loc[(y_exter['방송일시'].dt.month == mon)]
    
    d_exter = m_exter.groupby('방송일시').mean()
    d_exter.reset_index(inplace=True)

    # change datetime into string
    d_exter['방송일시'] = d_exter['방송일시'].dt.strftime('%m-%d %H')
    df_days['방송일시'] = df_days['방송일시'].dt.strftime('%m-%d %H')

    # Merge fix and external
    final_fix = pd.merge(df_days, d_exter, how='left', on='방송일시')
    final_fix.fillna(method='ffill', inplace=True)

    # Make categorical variable
    train_df = train_df[['노출(분)', '상품군', '판매단가', '상품구간_w',
                         '일시불_무이자', '일시불_할인율', '방송set', '상품명']]

    return final_fix, train_df


# Make dummy for categorical
def get_dum(fin_f, train_d):

    # dummy fix dataframe
    for i in range(4):
        fin_f = fin_f.append({'계절': i, '시간대': 0, '요일': 0},
                             ignore_index=True)

    dummy_fix = ['계절', '시간대', '요일']

    for dm in dummy_fix:
        fin_f[dm] = fin_f[dm].values.astype(int).tolist()
        fin_f[dm] = fin_f[dm].astype('category')

    fin_f = pd.get_dummies(fin_f, columns=dummy_fix)
    fin_f = fin_f.iloc[:-4, :]

    # dummy train dataframe
    train_d['상품군'] = train_d['상품군'].astype('category')
    train_d = pd.get_dummies(train_d, columns=['상품군'])

    return fin_f, train_d


# Split into Broad Set
def get_broad_group(df_train):
    broad_group = []

    for i in list(np.unique(df_train['방송set'])):
        df_braod_set = df_train[df_train['방송set'] == i]
        # Need all w2v columns
        broad_group.append(df_braod_set)

    return broad_group


def best_model(row):
    save_path = model_path+'xgb_bds18/'
    with open(save_path + 'model_0.pkl', 'rb') as f:
        model = pickle.load(f)
    result = model.predict(row)

    return result

# Get sales for broad_set


def calcu(b_g, day, before, after):
    n = len(b_g)
    days = [day] * n
    b_g_array = np.array(b_g)
    merge_array = []
    for i, j in zip(b_g_array, days):
        merge_array.append(list(i) + list(j))

    all_row = pd.DataFrame(merge_array)
    all_row.columns = before

    broad_set = all_row['방송set'].tolist()

    all_row.drop(['방송일시', '방송set'], axis=1, inplace=True)
    all_row_df = all_row[after]

    val = best_model(all_row_df.values)
    all_row_df['target'] = list(val)
    all_row_df['방송set'] = broad_set
    group_broad = all_row_df.groupby('방송set').mean()
    group_broad.sort_values(by=['target'], ascending=False, inplace=True)

    return list(group_broad.index)


# Final highest broad set index
def find_best_broad(broad_group, fix_time):
    #     results = pd.DataFrame()
    best_broad_ind = []
    best_scores = []
    broad_group_col = broad_group.columns.tolist()
    day_col = fix_time.columns.tolist()
    before_order = broad_group_col + day_col

    for i in tqdm_notebook(range(len(fix_time))):

        ind_best = calcu(broad_group, list(
            fix_time.iloc[i, :]), before_order, all_order)
        n = 0
        orders = 0
        while n == 0:
            if ind_best[orders] not in best_broad_ind:
                best_broad_ind.append(ind_best[orders])
                n += 1
            else:
                orders += 1

    return best_broad_ind


# Extract final dataframe
def final_df(fix_df, train_df, word_df, b_index):
    tr_w2v = pd.concat([train_df, word_df], axis=1)

    results = pd.DataFrame()

    for i, j in enumerate(b_index):
        broad = tr_w2v[tr_w2v['방송set'] == int(j)]
        temp_df = pd.DataFrame([fix_df.iloc[i, :]])
        rep_days = pd.concat([temp_df] * len(broad), ignore_index=True)
        rep_days.reset_index(inplace=True)
        broad.reset_index(inplace=True)
        result_df = pd.concat([rep_days, broad], axis=1)
        results = pd.concat([results, result_df], axis=0)

    return results[['방송일시', '노출(분)', '상품명', '상품군', '판매단가']]


def main(model_name='xgb_bds100'):
    # Make days dataframe
    # Input year and month. Enter two times
    df_days, mon = fixed_days()
    fix_df, train_df = fixed_all(mon)
    fix_df_d, train_df_d = get_dum(fix_df, train_df)
    
    # Extract just 1 week
    fix_df = fix_df.iloc[:126, :]
    fix_df_d = fix_df_d.iloc[:126, :]
    
    # Load word2vec
    with open(train_path + 'train_WordVec.pkl', 'rb') as fr:
        wordvec = pickle.load(fr)
    word_df = pd.DataFrame(
        wordvec, columns=['w2v{}'.format(i) for i in range(300)])
    
    # Load test result col
    with open(model_path + '%s/test_result.pkl'%(model_name), 'rb') as f_:
        test_result = pickle.load(f_)
    
    # Need order
    all_order = test_result['result']['data']['changed_x_col']
    
    # Merge train_df and word2vec_df
    train_w2v = pd.concat([train_df_d, word_df], axis=1)
    
    # Final best index
    best_index = find_best_broad(train_w2v, fix_df_d)
    
    # Final broadcast dataframe
    final_result = final_df(fix_df, train_df, word_df, best_index)
    
    # This is for predicting target
    final_result_2 = final_df(fix_df_d, train_df_d, word_df, best_index)
    target_pre = final_result_2[all_order]
    target_pred = best_model(target_pre.values)
    
    # Add target values
    final_result['취급액'] = list(target_pred)
    
    # Save broadcast result
    final_result.to_excel(output_path + '/cast/편성결과{0}월_{1}.xlsx'.format(mon, 1))

if __name__ =='__main__':
    main()