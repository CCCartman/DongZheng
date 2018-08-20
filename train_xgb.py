# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:05:38 2018

@author: rwhca
"""
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split

from get_both_columns import output
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X_train,X_test,y_train = output()

X_t,X_v,y_t,y_v = train_test_split(
    X_train, y_train, test_size=0.3, random_state=227)


def get_ntree():  
    f1_t_total, f1_v_total = [], []
    for ntree in range(10, 810, 10):
        xgb_base = XGBClassifier(objective='binary:logistic',n_estimators=ntree,
        random_state=1234,silent = 0,booster = 'gbtree',subsample = 0.8,
        colsample_bytree = 0.8,reg_alpha = 1,
        reg_lambda = 0,learning_rate = 0.1,
        max_depth = 6)
        
        print('此时 ntree = %s' % ntree)
        xgb_base.fit(X_t, y_t)
        y_t_pre = xgb_base.predict(X_t)
        y_v_pre = xgb_base.predict(X_v)
        f1_t_each = f1_score(y_t, y_t_pre,average = 'micro')
        f1_v_each = f1_score(y_v, y_v_pre,average = 'micro')
        f1_t_total.append(f1_t_each)
        f1_v_total.append(f1_v_each)
        myfile = open('D:\\workspace python\\contest\\accu_save\\' + 'xgbbase_810_1.txt',
                      'a', encoding='utf-8')
        print(f1_t_each,',',f1_v_each,file = myfile)
        myfile.close()
    return f1_t_total,f1_v_total


rmse_t_total,rmse_v_total = get_ntree()

f1_t_total,f1_v_total = [],[]
f = open('D:\\workspace python\\contest\\accu_save\\' + 'xgbbase_810_1.txt','r')
for line in f:
    line = line.split(',')
    f1_t_total.append(line[0])
    f1_v_total.append(line[1])
f.close()
f1_t_total = [float(i.rstrip()) for i in f1_t_total]
f1_v_total = [float(i.rstrip()) for i in f1_v_total]

### xgb 50树左右的位置最佳
plt.plot(range(10, 810, 10), f1_t_total,
         color='blue', marker='o',
         markersize=2, label='training f1-score')
plt.plot(range(10, 810, 10), f1_v_total,
         color='green', linestyle='--',
         marker='s', markersize=2,
         label='validation f1-score')
plt.grid()
plt.xlabel('Number of trees')
plt.ylabel('f1-score')
plt.legend(loc='best')
#plt.savefig('D:\\workspace python\\contest\\graph_save\\xgb_ntrees_800_best_50.png')
plt.show()


def tune_params():  
    f1_t_total, f1_v_total = [], []
    for max_depth in range(6,15):
        for subsample in [0.6,0.7,0.8]:
            for colsample_bytree in [0.6,0.7,0.8]:
                for reg_alpha in [0.1,1,10]:
                    xgb_base = XGBClassifier(objective = 'binary:logistic',n_estimators=50,
                                random_state=1234,silent = 0,booster = 'gbtree',subsample = subsample,
                                colsample_bytree = colsample_bytree,reg_alpha = reg_alpha ,
                                reg_lambda = 0,learning_rate = 0.1,n_jobs = 3,
                                max_depth = max_depth)
                    _params = { 'max_depth':max_depth,
                        'subsample':subsample,
                            'colsample_bytree':colsample_bytree,
                                'reg_alpha':reg_alpha,
                            }
                    xgb_base.fit(X_t, y_t)
                    y_t_pre = xgb_base.predict(X_t)
                    y_v_pre = xgb_base.predict(X_v)
                    f1_t_each = f1_score(y_t, y_t_pre,average = 'micro')
                    f1_v_each = f1_score(y_v, y_v_pre,average = 'micro')
                    f1_t_total.append(f1_t_each)
                    f1_v_total.append(f1_v_each)
                    print(_params)
                    myfile1 = open('D:\\workspace python\\contest\\accu_save\\' + 'xgbbase_saveparams_f1_0418.txt',
                                  'a', encoding='utf-8')
                    print(_params['max_depth'],_params['subsample'],_params['colsample_bytree'],
                          _params['reg_alpha'],file = myfile1)
                    
                    myfile1.close()
                    print(f1_t_each,f1_v_each)
                    myfile = open('D:\\workspace python\\contest\\accu_save\\' + 'xgbbase_tunparms_f1_0418.txt',
                                  'a', encoding='utf-8')
                    print(f1_t_each,',',f1_v_each,file = myfile)
                    myfile.close()                   
    return f1_t_total,f1_v_total

f1_t_total2,f1_v_total2 = tune_params()

file =  open('D:\\workspace python\\contest\\accu_save\\' + 'xgbbase_saveparams_f1_0418.txt',
                      'r', encoding='utf-8')
params_lst = []
for line in file:
    line = line.strip().split()
    params_lst.append(line)
file.close()
## 9个一组

plt.plot(range(243), f1_t_total2,
         color='blue', marker='o',
         markersize=1, label='training f1-score')
plt.plot(range(243), f1_v_total2,
         color='green', linestyle='--',
         marker='s', markersize=1,
         label='validation f1-score')
plt.grid()
plt.xlim([0,250])
plt.xlabel('params')
plt.ylabel('f1-score')
plt.legend(loc='best')
#plt.savefig('D:\\workspace python\\contest\\graph_save\\xgb_params.png')
plt.show()

f1_v_total2.index(max(f1_v_total2))
params_lst[25]
# '6', '0.8', '0.8', '1'


# xgb_best = XGBClassifier(objective = 'binary:logistic',n_estimators=50,
                                # random_state=1234,silent = 0,booster = 'gbtree',subsample = 0.8,
                                # colsample_bytree = 0.8,reg_alpha = 1 ,
                                # reg_lambda = 0,learning_rate = 0.1,n_jobs = 3,
                                # max_depth = 6,scale_pos_weight = 15)

# xgb_best.fit(X_t,y_t)
# joblib.dump(xgb_best,'D:\\workspace python\\contest\\model_save\\xgb_best.pkl')
xgb_best = joblib.load('D:\\workspace python\\contest\\model_save\\xgb_best.pkl')
y_v_pre = xgb_best.predict(X_v)
confusion_matrix(y_v,y_v_pre)
print(classification_report(y_v,y_v_pre))