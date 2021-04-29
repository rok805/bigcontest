# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:35:37 2020

@author: User
"""

from gensim.models.word2vec import Word2Vec
from itertools import chain

import pandas as pd
import numpy as np
import re
import pickle
import warnings
import utils

warnings.filterwarnings(action='ignore')


# Word2Vec
def word_vec(txt):
    model = Word2Vec(txt, size=300, window=1, workers=1, min_count=1)
    word_vectors = model.wv

    # Calculate average vector of each sentence
    final_vectors = []

    for i, j in enumerate(txt):
        vectors = []

        for k in j:
            try:
                vec = word_vectors[k]
                vectors.append(vec)
            except KeyError:
                print("Can't find vector: ", k)

        if len(vectors) == 0:
            print('Vectors length is 0: ', i)
        else:
            final_vectors.append(list(sum(vectors) / len(vectors)))

    return final_vectors


def txt_preprocessing(table, spacing):
    txt = table['상품명'].tolist()

    # Remove brakets
    spc = []

    for i in txt:
        i = re.sub('[\(\)\[\]\",*&!]', ' ', i)
        i = i.replace('  ', ' ')
        i = i.replace('   ', ' ')
        spc.append(i)

    # Tokenizing
    tok = []

    for i in spc:
        tok.append(i.split())

    # Save '1+1'
    token = []

    for i in tok:
        toke = []

        for j in i:
            if '1+1' in j:
                s = j.replace('1+1', '1+1 ')
                s = s.split()
                if len(s) == 2:
                    toke.append(s[0])
                    s[1] = re.sub('[+]', '', s[1])
                    toke.append(s[1])
                else:
                    toke.append(j)
            else:
                s = j.replace('+', ' ')
                s = s.split()
                if len(s) == 2:
                    toke.append(s[0])
                    toke.append(s[1])
                else:
                    if len(''.join(s)) != 0:
                        toke.append(''.join(s))

        token.append(toke)

    # Spacing
    space_token = []

    for t in token:
        space_t = []

        for i in t:
            try:
                change = spacing[1][spacing[0].index(i)].split()
                space_t.extend(change)
            except ValueError:
                space_t.append(i)

        space_token.append(space_t)

    # 다시 숫자 제거
    space_ttt = []

    for i in space_token:
        space_tt = []

        for j in i:
            if j not in ['F/W', 'S/S', '1+1']:
                a = re.sub('[^A-Za-z가-힣]', '', j)
                if len(a) == len(j):
                    if len(j) == 1:
                        if j not in ['x', 'D', 'W', '가',
                                     '그', '의', '분', '식']:
                            # To prevent removing words which length is one.
                            space_tt.append(j + 'a')
                    else:
                        space_tt.append(j)
            else:
                space_tt.append(j)

        space_ttt.append(space_tt)

    return space_ttt


train_path = utils.train_path
raw_path = utils.raw_path
processed_path = utils.processed_path


def main(train_file_name='', test_file_name=''):

    if train_file_name == '':
        train_file_name = 'train.csv'
    if test_file_name == '':
        test_file_name = '2020 빅콘테스트 데이터분석분야-챔피언리그_2020년 6월 판매실적예측데이터(평가데이터).xlsx'

    save_train_name = 'train_WordVec.pkl'
    save_test_name = 'test_WordVec.pkl'

    df = pd.read_csv(train_path+train_file_name)

    # Load test dataset
    df_test = pd.read_excel(raw_path+test_file_name)
    df_test.columns = df_test.iloc[0, :]
    df_test = df_test.iloc[1:, :]
    df_test = df_test[df_test['상품군'] != '무형']

    # Load spacing words
    with open(raw_path+'space_words.pickle', 'rb') as fr:
        space_words = pickle.load(fr)

    # Execute preprocessing function
    txt_train = txt_preprocessing(df, space_words)
    txt_test = txt_preprocessing(df_test, space_words)

    # Save vectors by train and test
    with open(processed_path+'txt_train.pickle', 'wb') as fw:
        pickle.dump(txt_train, fw)
    with open(processed_path+'txt_test.pickle', 'wb') as fw2:
        pickle.dump(txt_test, fw2)

    # Merge train text and test text
    txt_all = txt_train + txt_test

    # Execute word2vec function
    vectors = word_vec(txt_all)

    # Save vectors by train and test
    with open(train_path+save_train_name, 'wb') as fw:
        pickle.dump(vectors[:len(df)], fw)
    with open(train_path+save_test_name, 'wb') as fw2:
        pickle.dump(vectors[len(df):], fw2)


if __name__ == '__main__':
    main()
