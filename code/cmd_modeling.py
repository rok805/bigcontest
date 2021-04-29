# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 23:10:59 2020

@author: User
"""

import pandas as pd
import numpy as np
import os
import utils
from load_data import load_data




def main():
    
    config_path = utils.config_path
    count = 1
    name = 'model_setting.txt'
    for lr in np.arange(0.03,0.09,0.05):
        lr = np.round(lr,2)
        for max_depth in range(7,18,2):
            
            for gamma in np.arange(0,0.31,0.1):
                gamma = np.round(gamma,2)
                
                for subsample in np.arange(0.5,1.01,0.2):
                    subsample = np.round(subsample,2)
                    for colsample_bytree in np.arange(0.5,1.01,0.2):
                        colsample_bytree = np.round(colsample_bytree,2)
                        
                        for min_child_weight in np.arange(0.5,1.01,0.2):
                            min_child_weight = np.round(min_child_weight,2)
                            
                            for reg_alpha in range(0,3):
                                if count < 24:
                                    count +=1
                                    continue
                                
                                f1 = open(config_path+'default_setting.txt', 'r')
                                f2 = open(config_path+name,'w')
                                sw = 0
                                while True:
                                    line = f1.readline()
                                    
                                    if 'save_folder' in line:
                                        line = 'save_folder=xgb_bds%s\n'%(count)
                                    
                                    if 'xg' in line:
                                        sw = 1
                                    if sw == 1 and 'max_depth' in line:
                                        line = 'max_depth=%s\n'%(max_depth)
                                        
                                    elif sw == 1 and 'lr' in line:
                                        line = 'lr=%.2f\n'%(lr)
                                        
                                    elif sw == 1 and 'gamma' in line:
                                        line = 'gamma=%.2f\n'%(gamma)
                                        
                                    elif sw == 1 and 'subsample' in line:
                                        line = 'subsample=%.2f\n'%(subsample)
                                        
                                    elif sw == 1 and 'colsample_bytree' in line:
                                        line = 'colsample_bytree=%.2f\n'%(colsample_bytree)
                                    
                                    elif sw == 1 and 'min_child_weight' in line:
                                        line = 'min_child_weight=%.2f\n'%(min_child_weight)
                                    
                                    elif sw == 1 and 'reg_alpha' in line:
                                        line = 'reg_alpha=%s\n'%(reg_alpha)
                                    
                                    
                                    f2.write(line)
                                    
                                    if not line:
                                        break
                                
                                f1.close()
                                f2.close()
                                print('%s번째 시작'%count)
                                os.system('python modeling.py config/model_setting.txt')
                                count+=1


if __name__ == '__main__':
    main()
    


