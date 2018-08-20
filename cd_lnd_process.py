# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:15:50 2018

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

df = pd.read_csv(file_path + 'contest_ext_crd_cd_lnd.csv',encoding='utf-8')
df.columns

## 创建每个人开卡数
kaikashu = df.groupby('report_id')['loancard_id'].count()
kaikashu = pd.DataFrame(kaikashu)
kaikashu.columns = ['card_number']

## 数币种
bizhongshu = df.groupby('report_id')['currency'].apply(lambda x:x.drop_duplicates().count())
bizhongshu = pd.DataFrame(bizhongshu)
bizhongshu.columns = ['card_currency_number']

df2 = pd.merge(kaikashu,bizhongshu,left_index = True,right_index = True)
df2['report_id'] = df2.index.values.reshape(-1,1)

del kaikashu,bizhongshu

## credit_limit_amount
kadaikuanjine = df.groupby('report_id')['credit_limit_amount'].sum()
kadaikuanjine = pd.DataFrame(kadaikuanjine)
kadaikuanjine.columns = ['card_loan_amount']
df2 = pd.merge(df2,kadaikuanjine,left_on = 'report_id',right_index = True)

del kadaikuanjine

## 卡免担保比例
df.guarantee_type = df.guarantee_type.apply(lambda x:1 if x == '信用/免担保' else 0)
miandanbaoRate = df.groupby('report_id')['guarantee_type'].apply(lambda x:x.sum())/df.groupby('report_id')['guarantee_type'].count()
miandanbaoRate = pd.DataFrame(miandanbaoRate)
miandanbaoRate['report_id'] = miandanbaoRate.index.values.reshape(-1,1)
miandanbaoRate.columns = ['card_guarantee_rate','report_id']
df2 = pd.merge(df2,miandanbaoRate,on = 'report_id')

del miandanbaoRate

## 共享额度 ## 已用额度 ## 使用额度比例
gongxiangedu = df.groupby('report_id')['share_credit_limit_amount'].sum()
yiyongedu = df.groupby('report_id')['used_credit_limit_amount'].sum()
shiyongbili = yiyongedu / gongxiangedu



yiyongedu = pd.DataFrame(yiyongedu)
gongxiangedu = pd.DataFrame(gongxiangedu)
shiyongbili = pd.DataFrame(shiyongbili)

gongxiangedu.columns = ['card_share']
yiyongedu.columns = ['card_use']
shiyongbili.columns = ['card_use_rate']

df2 = pd.merge(df2,gongxiangedu,left_on = 'report_id',right_index = True)
df2 = pd.merge(df2,yiyongedu,left_on = 'report_id',right_index = True)
df2 = pd.merge(df2,shiyongbili,left_on = 'report_id',right_index = True)

del gongxiangedu,yiyongedu,shiyongbili
## 最近6个月平均使用额度 latest6_month_used_avg_amount
pingjunedu = df.groupby('report_id')['latest6_month_used_avg_amount'].sum()
pingjunedu = pd.DataFrame(pingjunedu)
pingjunedu.columns = ['card_average_6']
df2 = pd.merge(df2,pingjunedu,left_on = 'report_id',right_index = True)

del pingjunedu

## 最大使用额度 used_highest_amount
zuidaedu = df.groupby('report_id')['used_highest_amount'].sum()
zuidaedu  = pd.DataFrame(zuidaedu )
zuidaedu .columns = ['card_max_use']
df2 = pd.merge(df2,zuidaedu ,left_on = 'report_id',right_index = True)

del zuidaedu

## 本月应还款 本月实还款
yinghuankuan = df.groupby('report_id')['scheduled_payment_amount'].sum()
shihuankuan = df.groupby('report_id')['actual_payment_amount'].sum()
huankuanbili = shihuankuan / yinghuankuan

yinghuankuan = pd.DataFrame(yinghuankuan)
shihuankuan  = pd.DataFrame(shihuankuan)
huankuanbili = pd.DataFrame(huankuanbili)

yinghuankuan.columns = ['card_should_payment']
shihuankuan.columns = ['card_actpayment']
huankuanbili.columns = ['card_payment_rate']

df2 = pd.merge(df2,yinghuankuan,left_on = 'report_id',right_index = True)
df2 = pd.merge(df2,shihuankuan,left_on = 'report_id',right_index = True)
df2 = pd.merge(df2,huankuanbili,left_on = 'report_id',right_index = True)

## 还款日期

df['yanhoutianshu'] = pd.to_datetime(df.scheduled_payment_date) - pd.to_datetime(df.recent_pay_date)
df['yanhoutianshu'] = df['yanhoutianshu'].dt.days
yanhoutianshu =  df.groupby('report_id')['yanhoutianshu'].mean()
yanhoutianshu = pd.DataFrame(yanhoutianshu)
yanhoutianshu.columns = ['card_delay_payment']

df2 = pd.merge(df2,yanhoutianshu,left_on = 'report_id',
               right_index = True)

## 逾期期数 逾期金额

yuqiqishu = df.groupby('report_id')['curr_overdue_cyc'].sum()
yuqijine = df.groupby('report_id')['curr_overdue_amount'].sum()
df_temp = pd.DataFrame({'card_overdue_days':yuqiqishu,
                        'card_overdue_amount':yuqijine})
    
df2 = pd.merge(df2,df_temp,left_on = 'report_id',
               right_index = True)



## 第五部分 payment_state
payment_state_new = []
for pay_state in df.payment_state:
    if str(pay_state) != 'nan':
        if '1' in pay_state:
            pay_state = pay_state.replace('1','K' * int(np.exp(1)))
        if '2' in pay_state:
            pay_state = pay_state.replace('2','K' * int(np.exp(2)))
        if '3' in pay_state:
            pay_state = pay_state.replace('3','K' * int(np.exp(3)))
        if '4' in pay_state:
            pay_state = pay_state.replace('4','K' * int(np.exp(4)))
        if '5' in pay_state:
            pay_state = pay_state.replace('5','K' * int(np.exp(5)))
        if '6' in pay_state:
            pay_state = pay_state.replace('6','K' * int(np.exp(6)))
        payment_state_new.append(pay_state)
    else:
        payment_state_new.append(np.nan)
    

df.payment_state = payment_state_new

del payment_state_new
df['numofN'] = df.payment_state.str.count('K')

numofN = pd.DataFrame(df.groupby('report_id')['numofN'].sum())
numofN['report_id'] = numofN.index.values.reshape(-1,1)
numofN.columns = ['card_overdue','report_id']

df2 = pd.merge(df2,numofN,on = 'report_id')
del numofN

df2.to_csv(eda_path + 'cd_lnd_process.csv',index = False)


