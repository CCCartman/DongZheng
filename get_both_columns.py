# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 19:35:49 2018

@author: rwhca
"""
from make_train_data import make_train_data
from make_test_data import make_test_data

def make_train_test():
    df_train = make_train_data()
    df_test = make_test_data() 
    X_train,X_test,y = df_train.drop(['report_id','y'],axis = 1),df_test.drop(['report_id'],axis = 1),df_train.y
    
    return X_train,X_test,y

## 以train的列为准
def output():
    X_train,X_test,y_train = make_train_test()
    both_columns = []
    other_columns = []
    for col in X_test.columns.tolist():
        if col in X_train.columns.tolist():
            both_columns.append(col)
        else:
            other_columns.append(col)
    
    X_test = X_test.loc[:,both_columns]
    
    for col in other_columns:
        X_test[col] = 0
        
    print(X_train.shape,X_test.shape) 
    print(other_columns)
    return X_train,X_test,y_train

if __name__ == '__main__':
    output()
