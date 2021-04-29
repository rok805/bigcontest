2020.09.15

폴더 형식
* code 폴더 : py 파일 넣음
* data 폴더 : 각종 data를 넣음
  * raw 폴더 : train, test 원본 파일 등 전처리에 필요한 데이터
  * processed 폴더 : 1차 전처리 된 파일(시청률 데이터 등)
  * train 폴더 : train, test, word2vec 완성 파일이 있는 폴더


전처리 실행방법
* cmd 창에서 code 폴더로 간 다음에 'python processing_data.py' 하면 됨

데이터 로드하기  
from load_data import load_data  
load_data(data_name='', word_name='', word_option=True)
* data_name : train이나 test의 csv 파일 이름(확장자 포함)
* word_name : train이나 test word 파일의 pkl 이름(확장자 포함)
* word_option : True면 word_vector 붙여서, False면 붙이지 않고 반환함



참고

processing_train_test.py
* 사용법 : cmd 창에서 code 폴더로 간 다음에 'python processing_train_test.py' 하면 됨
* 저장하려는 파일 이름과

* 아래의 리스트의 경우 0번 index는 train, 1번 index는 test임
  * file_type_list : train, test 전처리 지칭하는 것(만지지 않아도 됨)
  * file_name_list : train, test 파일 이름
    * 폴더 안에 있는 파일 이름으로 뒤에 확장자까지 써줘야 함
    * 기본 형식은 다운 받았을 때 파일 이름
  * save_name_list
    * train, test 각각 저장하는 파일 이름


2019.09.16

processing_train_test.py
* 245~255번째 줄 사이에 무이자_일시불 변수 넣을 때 일시불 부분에서 target_product_list -> target_name_list으로 수정

2019.09.18

processing_stock_price.py
* 코스피 데이터를 받아서 전처리하는 함수

processing_train_test.py
* processing_kospi 함수 추가

2019.09.19

processing_train_test.py
* processing_kospi 함수 추가
* word2vec에 쓰일 key값을 여기서 넣는 것으로 수정
* outlier 처리부분 맨 마지막 줄에 추가
* train의 경우 outlier 삭제한 파일과 삭제하지 않은 파일 생성
  * train.csv : outlier 삭제한 파일
  * train_with_outlier.csv : outlier 삭제하지 않은 파일

processing_stock_price.py
* 안되던 부분 수정

load_data.py
* outlier 제거 한 파일과 제거 안 한 파일 둘 다 파일 이름 넣으면 열릴 수 있도록 수정.

processing_word2vec.py
* 무형제거하고, outlier 제거전 train 파일과 test파일 기준으로 단어 생성하도록 수정.

processing_data.py
* 이름 지정하는 부분 없애고 고정적인 이름만 나오도록 수정함.

2020.09.20
load_data.py
* df_per_b 함수 부분 수정 : 방송set으로 바꿈.

2020.09.22
processing_train_test.py
* change_column_name 함수 추가 및 적용
  * 각 변수 이름을 지정된 변수이름으로 바꿈
* processing_view 함수
  * 10분 단위 시청률 값 추가
  * 시청률 평균값을 하루 기준 02:00~01:59로 바꾸고 new_date값을 기준으로 merge 함
  * 변수를 생성하기 위해 make_str_holiday_weekend, make_str_just_week, make_str_min, make_hour_with_min 추가

model.py
* cross_validation 추가

modeling.py
* cross_validation 추가
* save시 모델 대신 예측값과 데이터의 인덱스값 저장으로 변경

2020.09.23
processing_train_test.py
* change_column_name : 변수명 변경하는 함수 최종 완성
* brandprime 생성하는 함수 삭제

load_data.py
* df_per_b : 중복된 함수 삭제 및 column 이름 변경
* load_data : word2vec embedding column 이름 수정

model.py
* train,valid,test로 나누도록 다시 수정

modeling.py
* train,valid,test로 나누도록 다시 수정
* 최종 mae, mape 값은 test 기준으로 만듦

2020.09.24
processing_train_test.py
* 최종 변수에서 ymd 변수 삭제 : 생성과정에 필요했던 변수였음

2020.09.25
processing_train_test.py
* processing_time 함수 추가 : 시간주기
* change_column_name : 잘못된 부분 수정
* 일시불_무이자 변수 생성 함수 삭제  
  -> 새로운 py파일에서 만듦

processing_data.py
* processing_make_var 추가 : 일시불_무이자 변수 생성  
  -> 추후 성별 등의 변수도 여기서 추가할 예정

processing_make_var.py
* 새로 넣은 py 파일로 일시불_무이자 관련 변수 생성

modeling.py, model.py
* 안되는 부분 등 수정. 

2020.09.26
utils.py
* fig_path, model_path 추가
  * fig_path : workspace/output/fig/
  * model_path : workspace/output/model/

processing_train_test.py
* processing_nocul_time 
  * w0.5_노출(분) 변수 추가  
    => 분석 후 삭제할 수도 있음
  * 잘못된 부분 수정
* processing_make_transport
  * 대중교통 관련 변수 만드는 함수
  * 위에 부속 함수들이 있음
  * 대중교통 날짜별로 csv 파일을 만들어 test 데이터도 사용할 수 있도록 만듦.
* processing_air
  * O3, 미세먼지 데이터 추가
