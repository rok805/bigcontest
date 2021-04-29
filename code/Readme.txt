2020.09.15

���� ����
* code ���� : py ���� ����
* data ���� : ���� data�� ����
  * raw ���� : train, test ���� ���� �� ��ó���� �ʿ��� ������
  * processed ���� : 1�� ��ó�� �� ����(��û�� ������ ��)
  * train ���� : train, test, word2vec �ϼ� ������ �ִ� ����


��ó�� ������
* cmd â���� code ������ �� ������ 'python processing_data.py' �ϸ� ��

������ �ε��ϱ�  
from load_data import load_data  
load_data(data_name='', word_name='', word_option=True)
* data_name : train�̳� test�� csv ���� �̸�(Ȯ���� ����)
* word_name : train�̳� test word ������ pkl �̸�(Ȯ���� ����)
* word_option : True�� word_vector �ٿ���, False�� ������ �ʰ� ��ȯ��



����

processing_train_test.py
* ���� : cmd â���� code ������ �� ������ 'python processing_train_test.py' �ϸ� ��
* �����Ϸ��� ���� �̸���

* �Ʒ��� ����Ʈ�� ��� 0�� index�� train, 1�� index�� test��
  * file_type_list : train, test ��ó�� ��Ī�ϴ� ��(������ �ʾƵ� ��)
  * file_name_list : train, test ���� �̸�
    * ���� �ȿ� �ִ� ���� �̸����� �ڿ� Ȯ���ڱ��� ����� ��
    * �⺻ ������ �ٿ� �޾��� �� ���� �̸�
  * save_name_list
    * train, test ���� �����ϴ� ���� �̸�


2019.09.16

processing_train_test.py
* 245~255��° �� ���̿� ������_�Ͻú� ���� ���� �� �Ͻú� �κп��� target_product_list -> target_name_list���� ����

2019.09.18

processing_stock_price.py
* �ڽ��� �����͸� �޾Ƽ� ��ó���ϴ� �Լ�

processing_train_test.py
* processing_kospi �Լ� �߰�

2019.09.19

processing_train_test.py
* processing_kospi �Լ� �߰�
* word2vec�� ���� key���� ���⼭ �ִ� ������ ����
* outlier ó���κ� �� ������ �ٿ� �߰�
* train�� ��� outlier ������ ���ϰ� �������� ���� ���� ����
  * train.csv : outlier ������ ����
  * train_with_outlier.csv : outlier �������� ���� ����

processing_stock_price.py
* �ȵǴ� �κ� ����

load_data.py
* outlier ���� �� ���ϰ� ���� �� �� ���� �� �� ���� �̸� ������ ���� �� �ֵ��� ����.

processing_word2vec.py
* ���������ϰ�, outlier ������ train ���ϰ� test���� �������� �ܾ� �����ϵ��� ����.

processing_data.py
* �̸� �����ϴ� �κ� ���ְ� �������� �̸��� �������� ������.

2020.09.20
load_data.py
* df_per_b �Լ� �κ� ���� : ���set���� �ٲ�.

2020.09.22
processing_train_test.py
* change_column_name �Լ� �߰� �� ����
  * �� ���� �̸��� ������ �����̸����� �ٲ�
* processing_view �Լ�
  * 10�� ���� ��û�� �� �߰�
  * ��û�� ��հ��� �Ϸ� ���� 02:00~01:59�� �ٲٰ� new_date���� �������� merge ��
  * ������ �����ϱ� ���� make_str_holiday_weekend, make_str_just_week, make_str_min, make_hour_with_min �߰�

model.py
* cross_validation �߰�

modeling.py
* cross_validation �߰�
* save�� �� ��� �������� �������� �ε����� �������� ����

2020.09.23
processing_train_test.py
* change_column_name : ������ �����ϴ� �Լ� ���� �ϼ�
* brandprime �����ϴ� �Լ� ����

load_data.py
* df_per_b : �ߺ��� �Լ� ���� �� column �̸� ����
* load_data : word2vec embedding column �̸� ����

model.py
* train,valid,test�� �������� �ٽ� ����

modeling.py
* train,valid,test�� �������� �ٽ� ����
* ���� mae, mape ���� test �������� ����

2020.09.24
processing_train_test.py
* ���� �������� ymd ���� ���� : ���������� �ʿ��ߴ� ��������

2020.09.25
processing_train_test.py
* processing_time �Լ� �߰� : �ð��ֱ�
* change_column_name : �߸��� �κ� ����
* �Ͻú�_������ ���� ���� �Լ� ����  
  -> ���ο� py���Ͽ��� ����

processing_data.py
* processing_make_var �߰� : �Ͻú�_������ ���� ����  
  -> ���� ���� ���� ������ ���⼭ �߰��� ����

processing_make_var.py
* ���� ���� py ���Ϸ� �Ͻú�_������ ���� ���� ����

modeling.py, model.py
* �ȵǴ� �κ� �� ����. 

2020.09.26
utils.py
* fig_path, model_path �߰�
  * fig_path : workspace/output/fig/
  * model_path : workspace/output/model/

processing_train_test.py
* processing_nocul_time 
  * w0.5_����(��) ���� �߰�  
    => �м� �� ������ ���� ����
  * �߸��� �κ� ����
* processing_make_transport
  * ���߱��� ���� ���� ����� �Լ�
  * ���� �μ� �Լ����� ����
  * ���߱��� ��¥���� csv ������ ����� test �����͵� ����� �� �ֵ��� ����.
* processing_air
  * O3, �̼����� ������ �߰�
