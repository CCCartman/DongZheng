# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 18:22:51 2018

@author: rwhca
"""

### --------------------读取相关模块--------------------
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from pylab import *  
from datetime import datetime
 
### --------------------模块读取完毕--------------------

### --------------------读取数据并探索性数据分析--------------------
### 工作路径
file_path = 'd:\\workspace python\\contest\\'
### EDA模块保存路径
eda_path = 'd:\\workspace python\\contest\\EDA\\'

df = pd.read_csv(file_path + 'contest_ext_crd_is_sharedebt.csv',encoding='utf-8')
df.columns

df.isnull().sum()

df2 = df.pivot_table(
        index=["REPORT_ID"],    
        columns=["TYPE_DW"],                 
        values=['FINANCE_CORP_COUNT FINANCE_ORG_COUNT','ACCOUNT_COUNT',
        'CREDIT_LIMIT','MAX_CREDIT_LIMIT_PER_ORG','MIN_CREDIT_LIMIT_PER_ORG'
        ,'BALANCE','USED_CREDIT_LIMIT','LATEST_6M_USED_AVG_AMOUNT']                     
        )

colnames = [(item[0],item[1][:-4]) for item in df2.columns.tolist()]
colnames = [''.join((item[1],item[0])) for item in colnames]

df_temp = pd.DataFrame(np.concatenate((df2.index.values.reshape(-1,1),df2.values),axis = 1))
df_temp.columns = ['REPORT_ID'] + colnames

df_temp.isnull().sum()
df_temp.dropna(thresh = 30000,axis = 1,inplace = True)

df_temp.to_csv(eda_path + 'sharedebt_process.csv',index = False)
