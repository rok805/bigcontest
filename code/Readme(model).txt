modeling.py, model.py

2020.09.21
modeling.py
* cmd���� ����� ������ ���� ������
  python modeling.py config/model_setting.txt
* config/model_setting.txt���� �Ķ���� ���� ����
  * ��ǥ�� ���е� ��� ���� ���� �� ��
  * setting : �⺻����, rf : ����������Ʈ, mlp : ��Ƽ�ۼ���, xg : xgboost
    * setting 
      * dummy : �������ڵ� ���·� ���� ������ (���� ��)
      * x_col : ������. ��ǥ�� ����
        * word embedding ������ �� �־ ��
      * y_col : ���Ӻ���(�� ������ ��)
      * word_option 
        * y : word embedding ���
        * n : �̻��
      * save_folder : output�������� ���������� ���� �̸�
      * train_valid_test : train, valid, test�� ���� ������(��ǥ ����)
      * normal_option
        * standard : z-score ������� ����ȭ
        * minmax : minmax ������� ����ȭ
        * none : ����ȭ ���� ����
      * save_name : save_folder�� �� �̸����� ������ ����
      * model
        * mlp : ��Ƽ�ۼ���
        * rg : ����������Ʈ
        * xgb : xgboost
      * random_state : ���� ��ȣ
      * cv_value : cross_validation ��. �� �� ��� 1�� �� ������ ��
      * train_file_name : train���� �� ������ �̸�(��ǰ�� ���� �н��ϴ� ��� �ƴϸ� train.csv�� ���� ��)
      * split_set_option : �н��� ���set������ �н� ����.
        * y : ���set���� �н�
        * n : ������ �н�
      
  * ������ ������ �� ���� �Ķ���� ����
  * early_stopping_rounds : 0���� �ϸ� early stopping ������ �ǹ�
  * output/save_folder/save_name�� ����� �����
    * �ڼ��� ������ ���� �߰�
* python���� for������ ���� ���氡���ϵ��� �� ����


model.py
* ���� ���� �ڵ�����
* mlp
* class Net���� hidden1,2,3,4,5�� �� hidden layer, ���� ���� ���ڴ� input node, output node��
  * �̸� �����ϸ鼭 �н���Ű�� ��


