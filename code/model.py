# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:42:44 2020

@author: User
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn.functional as F
import pdb
from xgboost import XGBRegressor
import random
import pickle
import shutil


def extract_data_set(dataset):
    X_train, y_train = dataset['train']
    X_valid, y_valid = dataset['valid']
    #X_test, y_test = dataset['test']
    return X_train, y_train, X_valid, y_valid



def score(model, X, y_true):
        
    if type(X) == torch.Tensor:
        y_pred = model(X).cpu().reshape(-1,)
        y_pred = y_pred.detach().numpy()
        y_true = y_true.cpu().numpy().reshape(-1,)
    else:
        y_pred = model.predict(X).reshape(-1,)
        y_true = y_true.reshape(-1,)
    
    # mae
    mae = np.mean(np.abs(y_pred-y_true))
    
    # mape
    non_zero_index = np.where(y_true>0)[0]
    y_pred_non_zero =  y_pred[non_zero_index]
    y_true_non_zero =  y_true[non_zero_index]
    mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero)/y_true_non_zero))
    
    return mae, mape


def cv_score(y_pred_list, y_true_list):
    mae_list = []
    mape_list = []
    for i in range(len(y_pred_list)):
        
        y_pred = y_pred_list[i].reshape(-1,)
        y_true = np.array(y_true_list[i]).reshape(-1,)
        mae = np.mean(np.abs(y_pred-y_true))
        
        # mape
        non_zero_index = np.where(y_true>0)[0]
        y_pred_non_zero =  y_pred[non_zero_index]
        y_true_non_zero =  y_true[non_zero_index]
        mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero)/y_true_non_zero))
        
        mae_list.append(mae)
        mape_list.append(mape)
        
    #pdb.set_trace()
    mae = np.mean(mae_list)
    mape = np.mean(mape_list)
    min_model_index = mape_list.index(min(mape_list))
    return mae, mape, min_model_index


def randomforest(dataset, config):
    X_train, y_train, X_valid, y_valid = extract_data_set(dataset)
    
    
    n_estimators = int(config['params']['n_estimators'])
    max_features = config['params']['max_features']
    if max_features not in ['auto','sqrt','log2']:
        max_features = float(max_features)
        
    max_depth = int(config['params']['max_depth'])
    min_samples_leaf = int(config['params']['min_samples_leaf'])
    random_state =int(config['params']['random_state'])
    
    regr = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, random_state=random_state, min_samples_leaf=int(len(X_train)*min_samples_leaf))
    regr.fit(X_train, y_train)
    feature_df = pd.DataFrame({'col':x_cols,'importance':np.round(regr.feature_importances_,4)})
    feature_df = feature_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    y_true = np.r_[y_valid,y_test]
    y_pred = np.r_[regr.predict(X_valid),regr.predict(X_test)]
    y_pred = np.array([i if i >=0 else 0 for i in y_pred])
    
    mae = np.mean(np.abs(y_pred-y_true))
    
    result_dict = {'feature_importance':feature_df,'dataset':dataset,
                   'y_pred':y_red,'y_true':y_true}
    
    return regr, result_dict


def mape_loss(predt, dtrain) :
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()   
    non_zero_index = np.where(y>0)[0]
    predt = predt[non_zero_index]
    y = y[non_zero_index]
    
    elements = np.abs((y-predt)/y)
    return 'MAPE', float(np.mean(elements))


def xgboosting(dataset, config, save_path):
    X_train_set, y_train_set, X_valid_set, y_valid_set = extract_data_set(dataset)

    n_estimators = int(config['xg']['n_estimators'])
    max_depth = int(config['xg']['max_depth'])
    lr = float(config['xg']['lr'])
    objective = config['xg']['objective']
    n_jobs = int(config['xg']['n_jobs'])
    gamma = float(config['xg']['gamma'])
    min_child_weight = float(config['xg']['min_child_weight'])
    subsample = float(config['xg']['subsample'])
    colsample_bytree= float(config['xg']['colsample_bytree'])
    reg_lambda = float(config['xg']['reg_lambda'])
    reg_alpha = float(config['xg']['reg_alpha'])
    cv_value = int(config['settings']['cv_value'])
    
    
    eval_metric = config['xg']['eval_metric']
    eval_metric = eval_metric.split(',')
    if eval_metric[0] == 'mape':
        eval_metric = mape_loss
    
    random_state =int(config['xg']['random_state'])
    early_stopping_rounds = int(config['xg']['early_stopping_rounds'])
    param_dist = {'objective':objective, 'n_estimators':n_estimators,
                  'learning_rate':lr,'max_depth':max_depth,
                  'n_jobs':n_jobs,
                  'gamma':gamma, 'min_child_weight':min_child_weight,
                  'subsample':subsample, 'colsample_bytree':colsample_bytree,
                  'reg_lambda':reg_lambda, 'reg_alpha':reg_alpha,
                  'random_state':random_state}
    
    y_test_pred_list = []
    y_test_true_list = []
    feature_list = []
    train_loss_list = []
    valid_loss_list = []
    min_model_index = 0
    for cv in range(cv_value):
        X_train, y_train, X_valid, y_valid = X_train_set[cv], y_train_set[cv],\
            X_valid_set[cv], y_valid_set[cv]
        
        #X_test, y_test = X_test_set[0], y_test_set[0]
        
        model = XGBRegressor(**param_dist)
        if early_stopping_rounds==0:
            model.fit(X_train,y_train,
                      verbose=True)
            
        else:
            eval_set = [(X_train,y_train),(X_valid, y_valid)]
            model.fit(X_train,y_train,eval_set=eval_set,
                      eval_metric=eval_metric,
                      verbose=True,early_stopping_rounds=early_stopping_rounds)
        
        feature = model.feature_importances_
        feature_list.append(feature)
        
        y_test_pred_list.append(model.predict(X_valid))
        y_test_true_list.append(y_valid)
        
        #pdb.set_trace()
        #pdb.set_trace()
        train_loss = model.evals_result()['validation_0']
        valid_loss = model.evals_result()['validation_1']
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        mae, mape = score(model, X_valid, y_valid)
        print('mae : %.3f, mape : %.3f'%(mae,mape))
        
        with open(save_path+'model_%s.pkl'%cv,'wb') as f:
            pickle.dump(model, f)
        model.save_model(save_path+'model_%s.json'%(cv))
        
    if cv_value <=1:
        mae, mape = score(model, X_valid, y_valid)
        
    else:
        
        mae, mape, min_model_index = cv_score(y_test_pred_list, y_test_true_list)
    
    result_dict = {'score':{'mae':mae, 'mape':mape, 'min_model_ind':min_model_index}, 
                   'feature':feature_list,
                   'pred':{'valid':y_test_pred_list},
                   'loss':{'train':train_loss,
                           'valid':valid_loss}}
    
    return result_dict




def lightbgm(dataset, config, save_path):
    X_train_set, y_train_set, X_valid_set, y_valid_set, X_test_set, y_test_set = extract_data_set(dataset)
    
    
    num_leaves = int(config['light']['num_leaves'])
    min_data_in_leaf = int(config['light']['min_data_in_leaf'])
    max_depth = int(config['light']['max_depth'])
    max_bin = int(config['light']['max_bin'])
    lr = float(config['light']['lr'])
    num_iterations = int(config['light']['num_iterations'])
    
    lambda_l1 = float(config['light']['lambda_l1'])
    lambda_l2 = float(config['light']['lambda_l2'])
    
    feature_fraction = float(config['light']['feature_fraction'])
    bagging_fraction = float(config['light']['bagging_fraction'])
    bagging_freq = int(config['light']['bagging_fractionbagging_freq'])
    
    early_stopping_rounds = int(config['light']['early_stopping_rounds'])
    objective = config['light']['objective']
    metric= config['light']['metric']
    
    n_jobs = int(config['light']['n_jobs'])
    cv_value = int(config['settings']['cv_value'])
    
    random_state =int(config['xg']['random_state'])
    early_stopping_rounds = int(config['xg']['early_stopping_rounds'])
    param_dist = {'objective':objective, 'n_estimators':n_estimators,
                  'learning_rate':lr,'max_depth':max_depth,
                  'n_jobs':n_jobs,
                  'min_child_weight':min_child_weight,
                  'max_delta_step':max_delta_step, 'max_delta_step':max_delta_step,
                  'reg_lambda':reg_lambda, 'reg_lambda':reg_lambda,
                  'seed':random_state}
    
    
    y_test_pred_list = []
    feature_list = []
    train_loss_list = []
    valid_loss_list = []
    min_model_index = 0
    for cv in range(cv_value):
        X_train, y_train, X_valid, y_valid = X_train_set[cv], y_train_set[cv],\
            X_valid_set[cv], y_valid_set[cv]
        
        X_test, y_test = X_test_set[0], y_test_set[0]
        
        model = XGBRegressor(**param_dist)
        if early_stopping_rounds==0:
            model.fit(X_train,y_train,
                      verbose=True)
            
        else:
            eval_set = [(X_train,y_train),(X_valid, y_valid)]
            model.fit(X_train,y_train,eval_set=eval_set,
                      eval_metric=eval_metric,
                      verbose=True,early_stopping_rounds=early_stopping_rounds)
        
        feature = model.feature_importances_
        feature_list.append(feature)
        
        y_test_pred_list.append(model.predict(X_test))
        
        
        #pdb.set_trace()
        #pdb.set_trace()
        train_loss = model.evals_result()['validation_0']
        valid_loss = model.evals_result()['validation_1']
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        mae, mape = score(model, X_test, y_test)
        print('mae : %.3f, mape : %.3f'%(mae,mape))
        
        with open(save_path+'model_%s.pkl'%cv,'wb') as f:
            pickle.dump(model, f)
        model.save_model(save_path+'model_%s.json'%(cv))
        
    if cv_value <=1:
        mae, mape = score(model, X_test, y_test)
        
    else:
        mae, mape, min_model_index = cv_score(y_test_pred_list, y_test_set)
        
    result_dict = {'score':{'mae':mae, 'mape':mape}, 'feature':feature_list,
                   'pred':{'test':y_test_pred_list},
                   'loss':{'train':train_loss,
                           'valid':valid_loss}}
    
    return result_dict



class Net(torch.nn.Module):
    def __init__(self, input_size, layer_list):

        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(input_size, layer_list[0])
        self.hidden2 = torch.nn.Linear(layer_list[0], layer_list[1])
        self.hidden3 = torch.nn.Linear(layer_list[1], layer_list[2])
        self.predict = torch.nn.Linear(layer_list[2], 1)   # output layer

    def forward(self, x):            
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)
        return x



def change_to_tensor(dataset):
    
    t_X_train, t_y_train, t_X_valid, t_y_valid = extract_data_set(dataset)
    X_train_list = []
    y_train_list = []
    X_valid_list = []
    y_valid_list = []


    for i in range(len(t_X_train)):
    
        X_train, y_train = t_X_train[i], t_y_train[i]
        X_valid, y_valid = t_X_valid[i], t_y_valid[i]
        
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float().reshape(-1)
        X_valid = torch.from_numpy(X_valid).float()
        y_valid = torch.from_numpy(y_valid).float().reshape(-1)
        
        if torch.cuda.is_available():
            cuda = torch.device('cuda')
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_valid = X_valid.cuda()
            y_valid = y_valid.cuda()
        
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_valid_list.append(X_valid)
        y_valid_list.append(y_valid)

    return X_train_list, y_train_list, X_valid_list, y_valid_list


def set_seed(random_state):
    torch.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_state)
    random.seed(random_state)
    random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar', add_name=''):
    torch.save(state, '%s/%s'%(save_path, filename+add_name))
    if is_best:
        shutil.copyfile('%s/%s'%(save_path, filename+add_name), 
                        '%s/model_best%s.pth.tar'%(save_path,add_name))


def calculate_mape(y_pred,y_true):
    
    nonzero_index = y_true.nonzero().view(-1)

    nonzero_y_true = y_true[nonzero_index]
    nonzero_y_pred = y_pred[nonzero_index]
    mape = torch.mean(torch.abs((nonzero_y_true-nonzero_y_pred)/nonzero_y_true)).item()
    return mape
    

def mlp(dataset, config, save_path):
    
    epochs = int(config['mlp']['epoch'])
    lr = float(config['mlp']['lr'])
    weight_decay = float(config['mlp']['weight_decay'])
    
    random_state = int(config['mlp']['random_state'])
    early_stopping_rounds= int(config['mlp']['early_stopping_rounds'])
    
    eval_metric= config['mlp']['eval_metric']
    cv_value = int(config['settings']['cv_value'])
    
    layer = config['mlp']['layer']
    layer1,layer2,layer3 = layer.split(',')
    layer1,layer2,layer3 = int(layer1),int(layer2),int(layer3)
    layer_list = [layer1,layer2,layer3]
    
    set_seed(random_state)
    
    X_train_list, y_train_list, X_valid_list, y_valid_list = change_to_tensor(dataset)
    #pdb.set_trace()
    
    train_loss_list = []
    valid_loss_list = []
    mae_list = []
    mape_list = []
    for cv in range(cv_value):
        X_train, y_train = X_train_list[cv], y_train_list[cv]
        X_valid, y_valid = X_valid_list[cv], y_valid_list[cv]
        
        n,m = X_train_list[0].shape
        model = Net(m,layer_list)
        if torch.cuda.is_available():
            cuda = torch.device('cuda')
            model = model.cuda()
            torch.cuda.device(0)
        criterion = torch.nn.MSELoss()
        """
        #  weight_decay : 값이 커질수록 가중치 값이 작어지게 되고, 
        오버피팅 현상을 해소할 수 있지만, weight_decay 값을 너무 크게 하면 
        언더피팅 현상이 발생하므로 적당한 값을 사용
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        early_count = 0
        min_value = 9999999999999999999999999
        
        min_epoch = 0
        print('모델링 시작')
        
        for epoch in range(epochs):
            
            is_best = False
            
            model.train()
            
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(X_train).reshape(-1)
            # Compute Loss
            train_loss = criterion(y_pred.squeeze(), y_train)
            #train_loss_value = y_pred.squeeze() - y_train
            #train_loss_list.append(train_loss)
            
            train_loss.backward()
            optimizer.step()
            
            # validation part
            model.eval()
            y_valid_pred = 0
            valid_loss = 0

            with torch.no_grad():
                y_valid_pred = model(X_valid).reshape(-1)
                valid_loss = criterion(y_valid_pred, y_valid)
                #valid_loss_list.append(valid_loss)
            

            # train mape, valid mape
            train_mape_loss = calculate_mape(y_pred, y_train)
            valid_mape_loss = calculate_mape(y_valid_pred, y_valid)
            
            
            # loss
            train_loss_list.append([train_loss.item(),train_mape_loss])
            valid_loss_list.append([valid_loss.item(),valid_mape_loss])
            
            # earlystopping
            if eval_metric == 'mape':
                target_value = valid_mape_loss
            else:
                target_value = valid_loss.item()
                
            if min_value > target_value:
                min_value = target_value
                early_count = 0
                min_epoch = epoch+1
                is_best = True
            
            else:
                early_count+=1
                
            print('Epoch {}: train loss: {} valid loss: {} | train mape : {} valid mape : {}'.format(epoch, train_loss.item(), valid_loss.item(), train_mape_loss,valid_mape_loss))
            
            # model_save
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': min_value,
                    'optimizer' : optimizer.state_dict(),
                },is_best, save_path, add_name=str(cv))
            
            
            if early_stopping_rounds !=0 and early_stopping_rounds == early_count:
                print('early stopping. epoch:%s'%(min_epoch))
                break
            
            
        # 평가
        model = Net(m,layer_list)
        if torch.cuda.is_available():
            cuda = torch.device('cuda')
            model = model.cuda()
            torch.cuda.device(0)
        criterion = torch.nn.MSELoss()
        checkpoint = torch.load('%s/model_best%s.pth.tar'%(save_path,cv))
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        mae, mape = score(model, X_valid, y_valid)
            
        mae_list.append(mae)
        mape_list.append(mape)
    
    f_mae = np.mean(mae_list)
    f_mape = np.mean(mape_list)
    
    result_dict = {'score':{'mae':f_mae, 'mape':f_mape}}
    result_dict['loss'] = {'train':train_loss, 'valid':valid_loss}
    
    return result_dict