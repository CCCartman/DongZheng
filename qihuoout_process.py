#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:09:54 2018

@author: yaomohan
"""

import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
#from read_data import readData

df = readData()



os.chdir('/Users/yaomohan/EDA/')
os.getcwd()


def readData():
    df = pd.read_csv('acontest_basic_train.csv',encoding = 'utf-8')
    df.columns = df.columns.str.lower()
    fileList = os.listdir('/Users/yaomohan/EDA/')
    fileList.remove('acontest_basic_train.csv')
    
    for file in fileList:
        df_temp = pd.read_csv(file,encoding = 'utf-8')
        df_temp.columns = df_temp.columns.str.lower()
        print(file)
        print('******合并前******')
        print(df_temp.shape)
        df = pd.merge(df,df_temp,left_on = 'report_id',
                      right_on = 'report_id',how = 'left')
        print('******合并后******')
        print(df.shape)

        del df_temp
    
    return df

df = readData()


def groupImmpute(df,col,switch = 'mean'):
    # val 0 val 1
    if switch == 'mean':
        val = df.groupby('y')[col].mean().tolist()
        
    if switch == 'median':
        val = df.groupby('y')[col].median().tolist()
    
    if switch == 'mode':
        val = df.groupby('y')[col].mode().tolist()
        
    idx0 = df[df[col].isnull() & (df.y == 0)].index
    idx1 = df[df[col].isnull() & (df.y == 1)].index
    
    df.loc[idx0,col] = val[0]
    df.loc[idx1,col] = val[1]
    
    return df[col]

df.sort_values(by='y',inplace = True)
## 变量名与变量格式修改
df.drop('id_card',axis = 1,inplace = True)

df.drop('loan_date',axis = 1,inplace = True)

df.drop('agent',axis = 1,inplace = True)

df.is_local = df.is_local.astype('category')

## 以后再处理了

#信用卡审批次数
df2 = df.copy()
df['card_approval_time'] = df[u'信用卡审批次数']
del df[u'信用卡审批次数']
groupImmpute(df,'card_approval_time',switch = 'mean')
#last_months
df2 = df.copy()
df['overdraft_last_months'] = df[u'last_months']
del df[u'last_months']
df[u'overdraft_last_months'].fillna(0,inplace=True)

#changing_amount
df[u'changing_amount'].fillna(0,inplace=True)

#未销户贷记卡latest_6m_used_avg_amount
df2 = df.copy()
df['card_unlogout_latest_6m_used_avg_amount'] = df[u'未销户贷记卡latest_6m_used_avg_amount']
del df[u'未销户贷记卡latest_6m_used_avg_amount']
#groupImmpute(df,'card_unlogout_latest_6m_used_avg_amount',switch = 'mean')
df[u'card_unlogout_latest_6m_used_avg_amount'].fillna(0,inplace=True)

#未销户贷记卡used_credit_limit
df2 = df.copy()
df['card_unlogout_used_credit_limit'] = df[u'未销户贷记卡used_credit_limit']
del df[u'未销户贷记卡used_credit_limit']
#groupImmpute(df,'card_unlogout_used_credit_limit',switch = 'mean')
df[u'card_unlogout_used_credit_limit'].fillna(0,inplace=True)

#未结清贷款latest_6m_used_avg_amount
df2 = df.copy()
df['loan_unsettle_latest_6m_used_avg_amount'] = df[u'未结清贷款latest_6m_used_avg_amount']
del df[u'未结清贷款latest_6m_used_avg_amount']
#groupImmpute(df,'loan_unsettle_latest_6m_used_avg_amount',switch = 'mean')
df[u'loan_unsettle_latest_6m_used_avg_amount'].fillna(0,inplace=True)

#未结清贷款balance
df2 = df.copy()
df['loan_unsettle_balance'] = df[u'未结清贷款balance']
del df[u'未结清贷款balance']
#groupImmpute(df,'loan_unsettle_balance',switch = 'mean')
df[u'loan_unsettle_balance'].fillna(0,inplace=True)

#未结清贷款account_count
df2 = df.copy()
df['loan_unsettle_account_count'] = df[u'未结清贷款account_count']
del df[u'未结清贷款account_count']
df[u'loan_unsettle_account_count'].fillna(0,inplace=True)

#未销户贷记卡account_count
df2 = df.copy()
df['card_unlogout_account_count'] = df[u'未销户贷记卡account_count']
del df[u'未销户贷记卡account_count']
df[u'loan_unsettle_account_count'].fillna(0,inplace=True)

#未销户贷记卡max_credit_limit_per_org'
df2 = df.copy()
df['card_unlogout_max_credit_limit_per_org'] = df[u'未销户贷记卡max_credit_limit_per_org']
del df[u'未销户贷记卡max_credit_limit_per_org']
df[u'card_unlogout_max_credit_limit_per_org'].fillna(0,inplace=True)

#未结清贷款min_credit_limit_per_org'
df2 = df.copy()
df['loan_unsettle_min_credit_limit_per_org'] = df[u'未销户贷记卡min_credit_limit_per_org']
del df[u'未销户贷记卡min_credit_limit_per_org']
df[u'loan_unsettle_min_credit_limit_per_org'].fillna(0,inplace=True)

#未结清贷款credit_limit'
df2 = df.copy()
df['loan_unsettle_credit_limit'] = df[u'未结清贷款credit_limit']
del df[u'未结清贷款credit_limit']
df[u'loan_unsettle_credit_limit'].fillna(0,inplace=True)

#未销户贷记卡credit_limit
df2 = df.copy()
df['card_unlogout_credit_limit'] = df[u'未销户贷记卡credit_limit']
del df[u'未销户贷记卡credit_limit']
df[u'card_unlogout_credit_limit'].fillna(0,inplace=True)

#'逾期金额'
df2 = df.copy()
df['overdraft_amount'] = df[u'逾期金额']
del df[u'逾期金额']
df[u'overdraft_amount'].fillna(0,inplace=True)

#out=df[['report_id','card_approval_time','overdraft_last_months','changing_amount','card_unlogout_latest_6m_used_avg_amount',
#        'card_unlogout_used_credit_limit','loan_unsettle_latest_6m_used_avg_amount','loan_unsettle_balance',
#        'loan_unsettle_account_count','card_unlogout_account_count','card_unlogout_max_credit_limit_per_org',
#        'loan_unsettle_min_credit_limit_per_org','loan_unsettle_credit_limit','card_unlogout_credit_limit',
#        'overdraft_amount']]