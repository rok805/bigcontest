# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:59:46 2020

@author: User
"""

import processing_word2vec
import processing_train_test
import processing_make_var

def main():

    processing_train_test.main()
    processing_word2vec.main()
    processing_make_var.main()
    print('완료. 화이팅!!!!!!')
    
if __name__ == '__main__':
    main()