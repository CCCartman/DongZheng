 # -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:27:55 2018

@author: rwhca
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.metrics import roc_curve,auc

os.chdir('D:\\workspace python\\contest\\')
from get_both_columns import output
download_filePath = 'D:\\workspace python\\contest\\model_save\\'

rf = joblib.load('D:\\workspace python\\contest\\model_save\\rf_best.pkl')
lgb = joblib.load('D:\\workspace python\\contest\\model_save\\lgb_best.pkl')
xgb = joblib.load('D:\\workspace python\\contest\\model_save\\xgb_best.pkl')

X_train,X_test,y_train = output()

X_t,X_v,y_t,y_v = train_test_split(
    X_train, y_train, test_size=0.3, random_state=227) 

    
def voting_model(X_train, X_test, y_train):    
    vclf = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('lgb',lgb)], 
                                        voting='soft', weights=[1,1,1])
    vclf.fit(X_train, y_train)
    predictions = vclf.predict_proba(X_test)[:,1]
    return predictions



def model_test(X_t,X_v,n1,n2,filepath = 'D:\\workspace python\\contest\\model_save\\'):
    res_t = np.zeros(3 * n1).reshape(3,n1)
    res_v = np.zeros(3 * n2).reshape(3,n2)
    xgb = joblib.load(filepath + 'xgb_best.pkl')
    lgb = joblib.load(filepath + 'lgb_best.pkl')
    rf = joblib.load(filepath + 'rf_best.pkl')
    clfs = [xgb,lgb,rf]
    for i,clf in enumerate(clfs):
        print('模型 %s 正在计算中' % clf)
        res_t[i] = clf.predict_proba(X_t)[:,1]
        res_v[i] = clf.predict_proba(X_v)[:,1]
    return res_t,res_v

res_t,res_v= model_test(X_t = X_t,X_v = X_v,n1 = X_t.shape[0],n2 = X_v.shape[0],
                        filepath = 'D:\\workspace python\\contest\\model_save\\')


res_t_df = pd.DataFrame(res_t.T,columns = ['xgb','lgb','rf'])
res_v_df = pd.DataFrame(res_v.T,columns = ['xgb','lgb','rf'])
## 调整三个模型加权平均的权重
f1_t_lst,f1_v_lst = [],[]
tune_lst = []
for i in np.arange(0.3,1,0.1):
    for j in np.arange(0.3,1,0.1):
        k = 1 - i -j
        if k > 0 :
            res_t_df['sum_'] = i*res_t_df.xgb + j*res_t_df.lgb + k*res_t_df.rf
            res_v_df['sum_'] = i*res_v_df.xgb + j*res_v_df.lgb + k*res_v_df.rf
            res_t_df['y_t_pre'] = res_t_df['sum_'].apply(lambda x:1 if x > 0.5 else 0)
            res_v_df['y_v_pre'] = res_v_df['sum_'].apply(lambda x:1 if x > 0.5 else 0)
            print(f1_score(y_t,res_t_df['y_t_pre'],average = 'micro'))
            print(f1_score(y_v,res_v_df['y_v_pre'],average = 'micro'))
            print(i,j,k)
            f1_t_lst.append(f1_score(y_t,res_t_df['y_t_pre'],average = 'micro'))
            f1_v_lst.append(f1_score(y_v,res_v_df['y_v_pre'],average = 'micro'))
            tune_lst.append((i,j,k))

plt.plot(range(len(f1_t_lst)),f1_t_lst,
         color = 'blue',marker = 'o',
         markersize = 2,label = 'training f1-score')  
plt.plot(range(len(f1_t_lst)), f1_v_lst,
         color='green', linestyle='--',
         marker='s', markersize=2,
         label='validation f1-score') 
plt.grid()
plt.xlabel('Weight Group')
plt.ylabel('f1-score')
plt.legend(loc='best')
plt.text(1,0.84,'(0.3,0.3,0.4)',ha='center',fontsize=11)
#plt.savefig('model2_weight.png')

vclf = VotingClassifier(estimators=[('xgb', xgb), ('lgb',lgb),('rf', rf)],  
                                        voting='soft', weights=[3,3,4])

vclf.fit(X_t,y_t)
joblib.dump(vclf,download_filePath + 'vclf.pkl')
y_v_pre_ori = vclf.predict_proba(X_v)[:,1]
y_v_pre = (1 - vclf.predict_proba(X_v)[:,1]) * 1000
y_v_pre = y_v_pre.astype(int)
y_v_pre_lb = np.where(y_v_pre < 500,1,0)
f1_score(y_v,y_v_pre_lb,average = 'micro')
confusion_matrix(y_v,y_v_pre_lb)
print(classification_report(y_v,y_v_pre_lb))


fpr,tpr,thresholds = roc_curve(y_v,y_v_pre_ori,
                               pos_label = 1)
plt.plot(fpr,tpr,lw=1,
         label = 'roc curve',color = 'red')
plt.plot([0,1],[0,1],
         linestyle = '--',
         color = (0.6,0.6,0.6),
         label = 'random guessing')
plt.plot([0,0,1],
         [0,1,1],
         lw = 2,linestyle = ':',color = 'black',
         label = 'perfect performance')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC曲线')
plt.legend(loc = 'best')
plt.show()

