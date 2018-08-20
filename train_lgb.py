# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:39:51 2018

@author: rwhca
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold 
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


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
        lgb_base = LGBMClassifier(n_estimators = ntree,objective = 'binary',
                      random_state=1234,n_jobs = 2,colsample_bytree=0.8, reg_alpha=1,
                      max_depth = 15, subsample = 0.8)

        print('此时 ntree = %s' % ntree)
        lgb_base.fit(X_t, y_t)
        y_t_pre = lgb_base.predict(X_t)
        y_v_pre = lgb_base.predict(X_v)
        f1_t_each = f1_score(y_t, y_t_pre,average = 'micro')
        f1_v_each = f1_score(y_v, y_v_pre,average = 'micro')
        f1_t_total.append(f1_t_each)
        f1_v_total.append(f1_v_each)
        myfile = open('D:\\workspace python\\contest\\accu_save\\' + 'lgbbase_810_2.txt',
                      'a', encoding='utf-8')
        print(f1_t_each,',',f1_v_each,file = myfile)
        myfile.close()
    return f1_t_total,f1_v_total


f1_t_total,f1_v_total = get_ntree()
# f1_t_total,f1_v_total =[],[]
f = open('D:\\workspace python\\contest\\accu_save\\' + 'lgbbase_810_2.txt','r')
for line in f:
    line = line.split(',')
    f1_t_total.append(line[0])
    f1_v_total.append(line[1])
f.close()
f1_t_total = [float(i.rstrip()) for i in f1_t_total]
f1_v_total = [float(i.rstrip()) for i in f1_v_total]

### lgb 150树
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
plt.savefig('D:\\workspace python\\contest\\graph_save\\lgb_ntrees_800.png')
plt.show()


def tune_params():  
    f1_t_total, f1_v_total = [], []
    for max_depth in range(6,15):
        for subsample in [0.6,0.7,0.8]:
            for colsample_bytree in [0.6,0.7,0.8]:
                for reg_alpha in [0.1,1,10]:
                    lgb_base = LGBMClassifier(n_estimators = 150,objective = 'binary',
                                      random_state=1234,n_jobs = 3,colsample_bytree=colsample_bytree, 
                                      reg_alpha=reg_alpha,
                                      max_depth = max_depth, subsample = subsample)
                    _params = { 'max_depth':max_depth,
                        'subsample':subsample,
                            'colsample_bytree':colsample_bytree,
                                'reg_alpha':reg_alpha,
                            }
                    lgb_base.fit(X_t, y_t)
                    y_t_pre = lgb_base.predict(X_t)
                    y_v_pre = lgb_base.predict(X_v)
                    f1_t_each = f1_score(y_t, y_t_pre,average = 'micro')
                    f1_v_each = f1_score(y_v, y_v_pre,average = 'micro')
                    f1_t_total.append(f1_t_each)
                    f1_v_total.append(f1_v_each)
                    print(_params)
                    myfile1 = open('D:\\workspace python\\contest\\accu_save\\' + 'lgbbase_saveparams_f1_0418.txt',
                                  'a', encoding='utf-8')
                    print(_params['max_depth'],_params['subsample'],_params['colsample_bytree'],
                          _params['reg_alpha'],file = myfile1)
                    
                    myfile1.close()
                    print(f1_t_each,f1_v_each)
                    myfile = open('D:\\workspace python\\contest\\accu_save\\' + 'lgbbase_tunparms_f1_0418.txt',
                                  'a', encoding='utf-8')
                    print(f1_t_each,',',f1_v_each,file = myfile)
                    myfile.close()                   
    return f1_t_total,f1_v_total

f1_t_total2,f1_v_total2 = tune_params()

file =  open('D:\\workspace python\\contest\\accu_save\\' + 'lgbbase_saveparams_f1_0418.txt',
                      'r', encoding='utf-8')
params_lst = []
for line in file:
    line = line.strip().split()
    params_lst.append(line)
file.close()


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
#plt.savefig('D:\\workspace python\\contest\\graph_save\\lgb_params.png')
plt.show()

params_lst[11]
# '6', '0.7', '0.6', '10'

#####################################
# lgb_best = LGBMClassifier(n_estimators = 150,objective = 'binary',
                                      # random_state=1234,n_jobs = 3,colsample_bytree=0.6, 
                                      # reg_alpha=10,learning_rate = 0.1,
                                      # max_depth = 6, subsample = 0.7,class_weight = {1:15,0:1})

# lgb_best.fit(X_t,y_t)
# joblib.dump(lgb_best,'D:\\workspace python\\contest\\model_save\\lgb_best.pkl')

lgb_best = joblib.load('D:\\workspace python\\contest\\model_save\\lgb_best.pkl')
y_v_pre = lgb_best.predict(X_v)
confusion_matrix(y_v,y_v_pre)
print(classification_report(y_v,y_v_pre))
######################################
 