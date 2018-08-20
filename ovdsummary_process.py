# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:26:12 2018

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

df = pd.read_csv(file_path + 'contest_ext_crd_is_ovdsummary.csv',encoding='utf-8')
df.columns
df.isnull().sum() # 不存在缺失
df.shape

df.columns = df.columns.str.lower()
df2 = df.pivot_table(
        index=['report_id'],   
        columns=['type_dw'],                
        values=['count_dw','months','highest_oa_per_mon','max_duration']              
     )


df2 = pd.DataFrame(np.concatenate((df2.index.values.reshape(-1,1),
                                   df2.values),axis = 1))
df2.columns = ['report_id','准贷记卡60天以上透支COUNT_DW','贷款逾期COUNT_DW','贷记卡逾期COUNT_DW',
               '准贷记卡60天以上透支MONTHS','贷款逾期MONTHS','贷记卡逾期MONTHS',
               '准贷记卡60天以上透支HIGHEST_OA_PER_MON','贷款逾期HIGHEST_OA_PER_MON','贷记卡逾期HIGHEST_OA_PER_MON',
               '准贷记卡60天以上透支MAX_DURATION','贷款逾期MAX_DURATION','贷记卡逾期MAX_DURATION'
              ]

df2.to_csv(eda_path + 'ovdsummary_process.csv',index = False)
df2.to_csv(eda_path + 'ovdsummary_process2.csv',index = False,encoding = 'UTF-8')