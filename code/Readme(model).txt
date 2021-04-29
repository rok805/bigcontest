modeling.py, model.py

2020.09.21
modeling.py
* cmd에서 실행시 다음과 같이 실행함
  python modeling.py config/model_setting.txt
* config/model_setting.txt에서 파라미터 수정 가능
  * 쉼표로 구분된 경우 띄어쓰기 하지 말 것
  * setting : 기본설정, rf : 랜덤포레스트, mlp : 멀티퍼셉션, xg : xgboost
    * setting 
      * dummy : 원핫인코딩 형태로 만들 데이터 (요일 등)
      * x_col : 설명변수. 쉼표로 구분
        * word embedding 변수는 안 넣어도 됨
      * y_col : 종속변수(안 만져도 됨)
      * word_option 
        * y : word embedding 사용
        * n : 미사용
      * save_folder : output폴더에서 하위폴더로 만들 이름
      * train_valid_test : train, valid, test로 나눌 비율값(쉼표 구분)
      * normal_option
        * standard : z-score 방식으로 정규화
        * minmax : minmax 방식으로 정규화
        * none : 정규화 적용 안함
      * save_name : save_folder에 모델 이름으로 저장할 변수
      * model
        * mlp : 멀티퍼셉션
        * rg : 랜덤포레스트
        * xgb : xgboost
      * random_state : 랜덤 번호
      * cv_value : cross_validation 값. 안 할 경우 1로 값 설정할 것
      * train_file_name : train으로 쓸 데이터 이름(상품군 나눠 학습하는 경우 아니면 train.csv로 쓰면 됨)
      * split_set_option : 학습시 방송set단위로 학습 여부.
        * y : 방송set단위 학습
        * n : 무작위 학습
      
  * 나머지 값들은 각 모델의 파라미터 값임
  * early_stopping_rounds : 0으로 하면 early stopping 미적용 의미
  * output/save_folder/save_name에 결과가 저장됨
    * 자세한 사항은 추후 추가
* python에서 for문으로 변수 변경가능하도록 할 예정


model.py
* 모델을 가진 코드파일
* mlp
* class Net에서 hidden1,2,3,4,5는 각 hidden layer, 안의 숫자 인자는 input node, output node임
  * 이를 조절하면서 학습시키면 됨


