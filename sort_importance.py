# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:24:57 2018

@author: Rui Wenhao
@Copyright: Rui Wenhao, All rights reserved
@Mail:rui_wenhao@cueb.edu.cn
"""
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from get_both_columns import output
import matplotlib.pyplot as plt
download_filePath = 'D:\\workspace python\\contest\\model_save\\'
os.chdir('D:\\workspace python\\contest\\')
X_train,X_test,y_train = output()

X_t,X_v,y_t,y_v = train_test_split(
    X_train, y_train, test_size=0.3, random_state=227) 

rf = joblib.load('D:\\workspace python\\contest\\model_save\\rf_best.pkl')
lgb = joblib.load('D:\\workspace python\\contest\\model_save\\lgb_best.pkl')
xgb = joblib.load('D:\\workspace python\\contest\\model_save\\xgb_best.pkl')
## 变量重要性排序

clfs = [rf,lgb,xgb]
def calcu_importance(clfs):
    imp = np.zeros(83)
    for clf in clfs:
        imp_temp = clf.feature_importances_/clf.feature_importances_.sum()
        imp += imp_temp
        del imp_temp
    return imp

def sort_importance(num = 10):
    cols = X_train.columns.tolist()
    imp = calcu_importance(clfs).tolist()
    imp_ = list(zip(cols,imp))
    imp_.sort(key = lambda x:x[1],reverse = True)
    
    print(imp_[:num])
    return imp_[:num]

imp_ = sort_importance()
imp_df = pd.DataFrame(imp_)
imp_df.columns = ['variable','score']
imp_df = imp_df.sort_values(by = 'score')
imp_df.loc[8,'variable'] = '6m_used_avg_amount'
imp_df.index = imp_df.variable

imp_df.plot.barh()