# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:26:17 2020

@author: User
"""


import pandas as pd
import pickle
import numpy as np
import re
import utils



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

def pre_processing_word(word_list, change_dict):
    for i in range(len(word_list)):
        word = word_list[i]
        if word in change_dict.keys():
            word = change_dict[word]
            
        word = word.replace('  ',' ')
        word_list[i] = word
    return word_list


def muiza_ilsibul_dict():
    change_name_dict = {}
    change_name_dict['(무) 한샘 하이바스 내추럴 기본형'] = '(무)한샘 하이바스 내추럴 기본형'
    change_name_dict['무) 한샘 하이바스 내추럴 기본형'] = '무)한샘 하이바스 내추럴 기본형'
    change_name_dict['무이자 대우전자 벽걸이 에어컨 TDOZ-S10JK'] = '무이자 대우전자 벽걸이 에어컨TDOZ-S10JK'
    change_name_dict['일시불 국내제조 삼성 4도어 푸드쇼케이스 냉장고 RH81R8011G2 (+청소기)']= '일시불 삼성 국내제조 4도어 푸드쇼케이스 냉장고 RH81R8011G2 (+청소기)'
    change_name_dict['무이자 삼성 무풍클래식 스탠드에어컨AF19R7573WZK'] = '무이자 삼성 무풍클래식 스탠드에어컨 AF19R7573WZK'
    change_name_dict['일시불 오스터 멀티 텀블러블랜더'] = '일시불 오스터 멀티 텀블러 블랜더'  
    change_name_dict['일시불 올리고 가스와이드그릴 프리미엄형'] = '일시불 올리고 가스와이드그릴레인지 프리미엄형'
    change_name_dict['무이자 해피콜 크로커다일 IH후라이팬'] = '무이자 해피콜 크로커다일 IH 후라이팬'
    change_name_dict['일시불 휴롬퀵스퀴저'] = '일시불 휴롬 퀵스퀴저'
    change_name_dict['일시불 20년 무풍 슬림 18형 화이트(절전) 스탠드(AF18T5774WZT) + 삼성 선풍기 2대(SFN-M35GABL)'] = '일시불 20년 무풍 슬림 18형 화이트(절전) 스탠드 (AF18T5774WZT) + 삼성 선풍기 2대(SFN-M35GABL)'
    change_name_dict['일시불 쿠첸 풀스텐 압력밥솥 10인용 (A1)'] = '일시불 쿠첸 풀스텐 압력밥솥 10인용(A1)'
    change_name_dict['무이자 삼성 UHD TVUN55RU7150FXKR'] = '무이자 삼성 UHD TV UN55RU7150FXKR'
    change_name_dict['무이자 삼성 UHD TVUN75RU7150FXKR'] = '무이자 삼성 UHD TV UN75RU7150FXKR'
    change_name_dict['무이자 올리고 가스와이드그릴레인지 프리미엄형'] = '무이자 올리고 가스와이드그릴 프리미엄형'
    change_name_dict['무이자 초특가 삼성5도어냉장고 T9000(RF84R9203S8) +시카고커트러리5p세트'] = '무이자 삼성5도어냉장고 T9000 (RF84R9203S8)'
    change_name_dict['일시불 삼성5도어냉장고 T9000 (RF84R9203S8) +공기청정기 (AX34R3020WWD)'] = '일시불 삼성5도어냉장고 T9000 (RF84R9203S8)'
    change_name_dict['무이자 쿠쿠전기밥솥 10인용(CRP-QS107FG/FS)'] = '무이자 쿠쿠전기밥솥 10인용'
    change_name_dict['일시불 쿠쿠전기밥솥 10인용 (QS)'] = '일시불 쿠쿠전기밥솥 10인용'
    change_name_dict['(삼성카드 6월 5%)무이자 LG 울트라HD TV 75UK6200KNB'] = '무이자 LG 울트라HD TV 75UK6200KNB'
    
    change_name_dict['무이자 삼성 UHDTV 55형 KU55UT7000FXKR+사운드바HW-T450'] = '무이자 삼성 UHDTV 55형 KU55UT7000FXKR'
    change_name_dict['무이자 삼성 UHDTV 75형  / KU75UT7000FXKR'] = '무이자 삼성 UHDTV 75형 KU75UT7000FXKR'
    change_name_dict['일시불 삼성 UHDTV 65형 KU65UT7000FXKR+삼성 사운드 바 / HW-T450'] = '일시불 삼성 UHDTV 65형 KU65UT7000FXKR'
    change_name_dict['일시불 삼성 UHDTV 75형  KU75UT7000FXKR + 삼성 사운드바HW-T450'] = '일시불 삼성 UHDTV 75형 KU75UT7000FXKR'
    change_name_dict['일시불 삼성 UHDTV 55형  KU55UT7000FXKR)'] = '일시불 삼성 UHDTV 55형 KU55UT7000FXKR'
    
    
    except_list = ['무이자 LG 휘센 에어컨 위너 FQ17V8WWJ1(스탠드)',
                   '무이자 에코바이런 호주 원목 대형 캄포도마 1종']
    
    return change_name_dict, except_list

def discriminate_unique_element(element):
    if len(element)!=1:
        return False
    else:
        return element[0]
    
    
def make_il_mu_dummy(df_, ok_word_df, type_='il'):
    if '일시불_무이자' not in df_.columns :
        df_['일시불_무이자'] = 0
    
    if type_=='il':
        df_.loc[ok_word_df.index,'일시불_무이자'] = 1
    elif type_=='mu':
        df_.loc[ok_word_df.index,'일시불_무이자'] = 2
    
    return df_



raw_path = utils.raw_path
train_path = utils.train_path
processed_path = utils.processed_path


def main():
    with open(raw_path+'space_words.pickle', 'rb') as f:
        word_vector = pickle.load(f)
    
    
    for df_name in ['train.csv','test.csv']:
    # 무이자,일시불 있는것만 따로 전처리
        dataframe = pd.read_csv(train_path+'%s'%(df_name))
        dataframe['상품명_n'] = dataframe['상품명']
        change_name_dict, except_list = muiza_ilsibul_dict()
        
        word = txt_preprocessing(dataframe, word_vector)
        word = [' '.join(i) for i in word]
        
        word_df = dataframe[['방송일시','상품명','마더코드','판매단가','방송set']]
        word_df['상품명n'] = word
        
        all_name_list = np.unique(word_df['상품명']).tolist()
        ok_mu_word_df = word_df[word_df['상품명n'].apply(lambda x: '무이자' in x)]
        ok_mu_word_df = ok_mu_word_df[['상품명','상품명n','마더코드','판매단가','방송set']]
        ok_mu_word_df['discriminate'] = 0
        
        ok_il_word_df = word_df[word_df['상품명n'].apply(lambda x: '일시불' in x)]
        ok_il_word_df = ok_il_word_df[['상품명','상품명n','마더코드','판매단가','방송set']]
        ok_il_word_df['discriminate'] = 1
        
        ok_word_df = pd.concat([ok_mu_word_df, ok_il_word_df], axis=0)
        
        
        # 무이자, 일시불 dummy 변수 형성
        dataframe = make_il_mu_dummy(dataframe, ok_il_word_df, 'il')
        dataframe = make_il_mu_dummy(dataframe, ok_mu_word_df, 'mu')
        
        
        #unique_mu = np.unique(ok_mu_word_df['상품명n'], return_counts=True)
        ok_word_df['changed_상품명'] = pre_processing_word(ok_word_df['상품명'].tolist(), change_name_dict)
        
        
        # 사전 정제
        #mu_word = pre_processing_word(mu_word, change_name_dict)
        #il_word = pre_processing_word(il_word, change_name_dict)      
        
        
        
        discount_ratio_list = np.array([0]*len(dataframe))
        discount_ratio_list = discount_ratio_list.astype('float64')
        discount_list = np.array([0]*len(dataframe))
        
        
        check_list = np.unique(ok_word_df[ok_word_df['discriminate']==0]['changed_상품명']).tolist()
        
        key_dict={}
        for check_w in check_list:

            ok_check_w_df = ok_word_df[ok_word_df['changed_상품명']==check_w]
            target_set_list = np.unique(ok_check_w_df['방송set'])
            
            target_temp_df = ok_word_df[ok_word_df['방송set'].isin(target_set_list)]
            target_product_set = np.unique(target_temp_df['changed_상품명']).tolist()
            
            set_list = np.unique(ok_check_w_df['방송set'])
            
            target_w = ''
            if '무이자' in check_w:
                target_w = check_w.replace('무이자', '일시불')
            elif '무' in check_w:
                target_w = check_w.replace('무', '일')
            
            # 할인율 : (일시불/무이자) - 1
            if target_w in target_product_set:
                target_temp_df = target_temp_df[['상품명','changed_상품명','방송set','판매단가']]
                
                for set_num in set_list:
                    set_temp_df = target_temp_df[target_temp_df['방송set']==set_num]
                    check_w_df = set_temp_df[set_temp_df['changed_상품명']==check_w]
                    target_w_df = set_temp_df[set_temp_df['changed_상품명']==target_w]
                    
                    check_price = np.unique(check_w_df['판매단가'])
                    check_price = discriminate_unique_element(check_price)
                    if check_price == False:
                        continue
                    target_price = np.unique(target_w_df['판매단가'])
                    target_price = discriminate_unique_element(target_price)
                    if target_price == False:
                        continue
                    
                    discount_ratio = (target_price/check_price)-1
                    discount_price = target_price - check_price
                    
                    check_index = check_w_df.index.tolist()        
                    target_index = target_w_df.index.tolist()
                    
                    # 처리
                    discount_ratio_list[target_index] = discount_ratio
                    discount_list[target_index] = discount_price
                    
                    
                    dataframe['상품명_n'].loc[check_index] = check_w
                    dataframe['상품명_n'].loc[target_index] = target_w
                    
                    
                    key = check_w
                    value = target_w
                    
                    key_dict[key]=value
                    
            elif check_w in except_list:
                continue
            else:
                raise ValueError
        
        with open(processed_path+'%s_mu_il.pkl'%(df_name),'wb') as f:
            pickle.dump(key_dict,f)
            
            
        dataframe['일시불_할인율'] = discount_ratio_list
        dataframe['일시불_할인액'] = discount_list

        
        dataframe.to_csv(utils.train_path+'%s'%(df_name), index=False)