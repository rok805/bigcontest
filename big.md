## 코드 내용 정리
EDA 및 전처리 -> 모델생성 -> 예측 및 모델 분석 -> 편성표 생성 순으로 구성


## 코드 기본 실행 방법
* cmd 창에서 workspace/code로 이동
* python processing_data.py
  * 전처리
* python modeling.py config/model_setting.txt
  * 예측 모델 만듦
  * xgboost가 가능하며, 하이퍼 파라미터는 config/model_setting.txt에서 변경 가능



#### 이상치 EDA.ipynb
* 2019년 실적 데이터의 이상치를 정의하기 위해 사용한 코드
* 최종적인 이상치 범위를 정함

#### processing_data.py
* processing_train_test.py, processing_word2vec.py, processing_make_var.py를 한번에 실행할 수 있도록 만드는 전처리 함수

#### processing_train_test.py
* 각종 변수를 전처리하여 분석 단계의 train, test 파일을 만드는 코드
* 전처리 변수에 따라 함수를 나누어 다른 변수 전처리시 해당 변수의 함수만 추가 가능

#### processing_word2vec.py
* 실적 데이터의 상품명을 기준으로 워드임베딩을 하는 코드
* 각 상품명을 단어 단위로 미리 전처리하여 만든 파일을 활용

#### processing_make_var.py
* 일시불_무이자 관련 변수를 생성하는 코드
* 단어 사전과 실적의 상품명을 활용하여 변수 생성함

#### 취급액 EDA.ipynb
* 2019년 실적 데이터의 취급액 분포를 보기 위한 코드

#### 방송구간_w 변수.ipynb
* 방송별 구간에 따른 취급액의 패턴을 분석하기 위한 코드

#### 시간주기변수.ipynb
* 24시간을 기준으로 시간에 따라 취급액의 패턴이 있는지 분석하는 코드

#### load_data.py
* train,test 파일을 load 할 때 쓰는 코드
* 방송set 단위, colab용, local용으로 나눔

#### utils.py
* 자주 사용하는 경로, 함수를 저장한 코드

#### make_air_pollution.py
* 미세먼지 데이터를 생성하는 코드

#### eda_make_data.py
* 대중교통, 방송 데이터, 무이자_일시불, 대중교통, 미세먼지에 대한 EDA를 하는 코드

#### modeling.py
* 예측 모델을 만들 때 train과 valid를 나누는 등 모델을 만들기 위한 준비 코드
* standard 등 다양한 옵션 적용 가능

#### cmd_modeling.py
* 하이퍼 파라미터를 바꾸면서 modeling.py를 실행 할 때 쓰는 코드

#### model.py
* xgboost, MLP 등 모델 구조를 가진 코드

#### predict.py
* 만든 모델로 결과를 예측하는 코드
* test 모델을 기본으로 하며, 예측값 및 feature importance 그래프 반환

#### broadcast.py
* 위에서 만든 예측 모델을 기반으로 최적의 월간 방송 편성 시스템을 구축하는 코드로 예측 취급액까지 제공함
* 학습 모델을 기반으로 자동으로 만들기 때문에 데이터를 많이 넣을수록 예측 취급액의 정확도가 높아짐