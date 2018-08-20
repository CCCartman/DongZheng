# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:35:40 2018

@author: Rui Wenhao
@Copyright: Rui Wenhao, All rights reserved
@Mail:rui_wenhao@cueb.edu.cn
"""

import pickle
import pandas as pd
from get_both_columns import output
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
 
X_train,X_test,y_train = output()
download_filePath = 'D:\\workspace python\\contest\\model_save\\'

vclf = joblib.load(download_filePath + 'vclf.pkl')

y_test = (1 - vclf.predict_proba(X_test)[:,1]) * 1000
y_test = y_test.astype(int)
y_test = pd.DataFrame(y_test,columns = ['score'])

test = pd.read_csv('D:\\workspace python\\contest\\EDA\\testDocument\\contest_basic_test.csv')
df = pd.concat([test.REPORT_ID,y_test],axis = 1)
del test,y_test

df['y'] = df.score.apply(lambda x:1 if x < 500 else 0)

df.to_csv('D:\\workspace python\\contest\\accu_save\\y_predict.csv',index = False)
def draw_y(df):
    '''
    绘制响应变量y的条形图观察分布情况
    '''
    diagnosis_text = ['不违约','违约'] 
    plt.bar(range(2),df.y.value_counts(),color=['bisque','pink'])
    plt.xticks((0,1),diagnosis_text,fontsize=15)
    plt.xlabel('是否违约',fontsize=15)
    plt.ylabel('频数',fontsize=15)
    plt.title('违约情况条形图',fontsize=20)
    for i,value in enumerate(df.y.value_counts()):
        plt.text(i,value-10, '%s'%value,ha='center',fontsize=15)
    plt.show()
    
draw_y(df)

df[df.score == df.score.max()]
df[df.score == df.score.min()]
df.score.describe()
df.score.hist(bins = 15)
