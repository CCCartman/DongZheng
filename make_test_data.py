# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 23:04:17 2018

@author: rwhca
"""

import pandas as pd
from process_after_combine import processData

## 把上一个模块导入的函数 封装到一个新的函数里才可以用

def read_test_data(filePath,file_1):
    df = processData(filePath,file_1,train = False)
    return df


def make_test_data():
    df = read_test_data(filePath = 'D:\\workspace python\\contest\\EDA\\testDocument\\',file_1 = 'contest_basic_test.csv')
    colnames_ = ['is_local','edu_level','marry_status',
                 'has_fund','ln_abnormal','salary']
    new_cols = pd.get_dummies(df[colnames_])
    
    df_temp = df.loc[:,['report_id']]
    
    df.drop(['edu_level','marry_status','has_fund','ln_abnormal','salary',
             'is_local','ln_payment_ontime','ln_payment_random','report_id',
             'announce_count','dissent_count'],axis=1,inplace = True)
    
    df = df.apply(lambda x: (x-x.mean())/(x.std() + 0.0000001),axis = 0)
    df = df.join(new_cols)
    df = pd.concat([df,df_temp],axis = 1)
    return df

