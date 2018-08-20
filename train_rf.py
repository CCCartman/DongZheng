# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 20:21:27 2018

@author: rwhca
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

from get_both_columns import output
import time
import pickle
import matplotlib.pyplot as plt
X_train,X_test,y_train = output()

X_t,X_v,y_t,y_v = train_test_split(
    X_train, y_train, test_size=0.3, random_state=227)
   

def get_ntree():  
    f1_t_total, f1_v_total = [], []
    for ntree in range(10, 500, 10):
        rf_base = RandomForestClassifier(n_estimators = ntree,
                                        random_state=1234,max_depth = 8,n_jobs = 4)
        print('此时 ntree = %s' % ntree)
        rf_base.fit(X_t, y_t)
        y_t_pre = rf_base.predict(X_t)
        y_v_pre = rf_base.predict(X_v)
        f1_t_each = f1_score(y_t, y_t_pre,average = 'micro')
        f1_v_each = f1_score(y_v, y_v_pre,average = 'micro')
        f1_t_total.append(f1_t_each)
        f1_v_total.append(f1_v_each)
        myfile = open('D:\\workspace python\\contest\\accu_save\\' + 'rfbase_810_4.txt',
                      'a', encoding='utf-8')
        print(f1_t_each,',',f1_v_each,file = myfile)
        myfile.close()
    return f1_t_total,f1_v_total

f1_t_total,f1_v_total = get_ntree()

f1_t_total,f1_v_total = [],[]
f = open('D:\\workspace python\\contest\\accu_save\\' + 'rfbase_810_2.txt','r')
for line in f:
    line = line.split(',')
    f1_t_total.append(line[0])
    f1_v_total.append(line[1])
f.close()
f1_t_total = [float(i.rstrip()) for i in f1_t_total]
f1_v_total = [float(i.rstrip()) for i in f1_v_total]

## ntree = 40 f1 = 0.938555
plt.plot(range(10,500,10), f1_t_total,
         color='blue', marker='o',
         markersize=2, label='training f1-score')
plt.plot(range(10,500,10), f1_v_total,
         color='green', linestyle='--',
         marker='s', markersize=2,
         label='validation f1-score')
plt.grid()
plt.xlabel('Number of trees')
plt.ylabel('f1-score')
plt.legend(loc='best')
#plt.savefig('D:\\workspace python\\contest\\graph_save\\lgb_ntrees_800.png')
plt.show()

def tune_params(): 
    params_lst = []
    f1_t_total,f1_v_total = [], []
    for max_features in range(30,70,10):
        for max_depth in range(6,20):
            rf_base = RandomForestClassifier(n_estimators = 10,
                                        random_state=1234,max_depth = max_depth,n_jobs = 4)
            _params = {
                'max_features':max_features,
                'max_depth':max_depth
                    }
            rf_base.fit(X_t, y_t)
            y_t_pre = rf_base.predict(X_t)
            y_v_pre = rf_base.predict(X_v)
            f1_t_each = f1_score(y_t, y_t_pre,average = 'micro')
            f1_v_each = f1_score(y_v, y_v_pre,average = 'micro')
            f1_t_total.append(f1_t_each)
            f1_v_total.append(f1_v_each)
            print(_params)
            myfile1 = open('D:\\workspace python\\contest\\accu_save\\' + 'rfbase_tunparms_f1_0419.txt',
                          'a', encoding='utf-8')
            print([_params['max_features'],_params['max_depth']],file = myfile1)   
            params_lst.append([_params['max_features'],_params['max_depth']])
            myfile1.close()
            print(f1_t_each,f1_v_each)
            myfile = open('D:\\workspace python\\contest\\accu_save\\' + 'rfbase_tunparms_f1_0419.txt',
                          'a', encoding='utf-8')
            print(f1_t_each,',',f1_v_each,file = myfile)
            myfile.close()                   
    return params_lst,f1_t_total,f1_v_total


params_lst,f1_t_total2,f1_v_total2 = tune_params()

# max_depth = 14
plt.plot(range(56), f1_t_total2,
         color='blue', marker='o',
         markersize=3, label='training f1-score')
plt.plot(range(56), f1_v_total2,
         color='green', linestyle='--',
         marker='s', markersize=3,
         label='validation f1-score')
plt.grid()
plt.xlabel('params')
plt.ylabel('f1-score')
plt.legend(loc='best')
#plt.savefig('D:\\workspace python\\contest\\graph_save\\lgb_params.png')
plt.show()

# params_lst[5] # 30 17

# rf_best = RandomForestClassifier(n_estimators = 10,max_features =30,
                                    # random_state=1234,max_depth = 11,n_jobs = 4,
                                   # class_weight = {1:15,0:1})

# rf_best.fit(X_t,y_t)
joblib.dump(rf_best,'D:\\workspace python\\contest\\model_save\\rf_best.pkl')

rf_best = joblib.load('D:\\workspace python\\contest\\model_save\\rf_best.pkl')
y_v_pre = rf_best.predict(X_v)
confusion_matrix(y_v,y_v_pre)
print(classification_report(y_v,y_v_pre))
