# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 12:51:20 2018

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

### --------------------分析contest_ext_crd_cd_ln.csv--------------------

df = pd.read_csv(file_path + 'contest_ext_crd_cd_ln.csv',encoding='gbk'
                 ,sep = '\t')
df.columns
df
## 第一部分 jieqingRate
jieqingRate = df[df.state == '结清'].groupby('report_id')['credit_limit_amount'].sum()/df.groupby('report_id')['credit_limit_amount'].sum()
jieqingRate = pd.DataFrame(jieqingRate)
jieqingRate['report_id'] = jieqingRate.index.values.reshape(-1,1)
jieqingRate.fillna(0,inplace = True)  # 0就是没还清
jieqingRate.columns = ['贷款结清比例','report_id']

## 第二部分
df.state = df.state.str.replace('呆账',
                                '异常').str.replace('逾期',
                                    '异常').str.replace('转出','异常')

df['是否异常'] = df.state.apply(lambda x:1 if x == '异常' else 0)
shifouyichang = df.groupby('report_id')['是否异常'].sum()
shifouyichang = shifouyichang.apply(lambda x:1 if x > 0 else 0)
shifouyichang = pd.DataFrame(shifouyichang)
shifouyichang['report_id'] = shifouyichang.index.values.reshape(-1,1)
shifouyichang.columns = ['贷款是否异常','report_id']

df2 = pd.merge(jieqingRate,shifouyichang,on = 'report_id')
del jieqingRate,shifouyichang

## 第三部分 处理担保方式
df.guarantee_type.value_counts()
df.guarantee_type = df.guarantee_type.apply(lambda x:1 if x == '信用/免担保' else 0)
miandanbaoRate = df.groupby('report_id')['guarantee_type'].apply(lambda x:x.sum()/x.count())
miandanbaoRate = pd.DataFrame(miandanbaoRate)
miandanbaoRate['report_id'] = miandanbaoRate.index.values.reshape(-1,1)
miandanbaoRate.columns = ['贷款免担保比例','report_id']
df2 = pd.merge(df2,miandanbaoRate,on = 'report_id')

del miandanbaoRate
## 第四部分 paymentRating
def huanqianTime(df):
    df_temp = df.loc[:,['report_id','payment_rating','payment_cyc']]
    df_temp.loc[df.payment_rating == '按季归还','payment_cyc'] = df_temp.loc[df.payment_rating == '按季归还','payment_cyc'] * 3
    df_temp.loc[df.payment_rating == '按日归还','payment_cyc'] = df_temp.loc[df.payment_rating == '按日归还','payment_cyc'] // 30
    df_temp.loc[df.payment_rating == '按年归还','payment_cyc'] = df_temp.loc[df.payment_rating == '按年归还','payment_cyc'] * 12
    df_temp.loc[df.payment_rating == '按半年归还','payment_cyc'] = df_temp.loc[df.payment_rating == '按半年归还','payment_cyc'] * 6
    df_temp.loc[df.payment_rating == '按周归还','payment_cyc'] = df_temp.loc[df.payment_rating == '按周归还','payment_cyc'] //4 
    
    df_temp.loc[df.payment_rating == '按季归还','payment_rating'] = '按周期归还'
    df_temp.loc[df.payment_rating == '按日归还','payment_rating'] ='按周期归还'
    df_temp.loc[df.payment_rating == '按年归还','payment_rating'] = '按周期归还'
    df_temp.loc[df.payment_rating == '按半年归还','payment_rating'] = '按周期归还'
    df_temp.loc[df.payment_rating == '按周归还','payment_rating'] = '按周期归还'
    df_temp.loc[df.payment_rating == '按月归还','payment_rating'] = '按周期归还'
    
    return df_temp

huanqian = huanqianTime(df)
huanqian = huanqian.pivot_table(
   index=["report_id"],
   columns=["payment_rating"],               
    values=["payment_cyc"],
    aggfunc=[np.sum]
    )
huanqian = pd.DataFrame(np.concatenate((huanqian.index.values.reshape(-1,1),huanqian.values),axis = 1))
huanqian.columns = ['report_id','贷款一次性归还','贷款不定期归还','贷款按其他方式归还','贷款按周期归还']
huanqian.fillna(0,inplace = True)

df2 = pd.merge(df2,huanqian,on = 'report_id')

del huanqian

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
numofN.columns = ['贷款逾期程度','report_id']

df2 = pd.merge(df2,numofN,on = 'report_id')
del numofN

## 第五部分 credit_limit_amount、balance
hetongjine = df.groupby('report_id')['credit_limit_amount'].sum()
benjinyue = df.groupby('report_id')['balance'].sum()
yuebili = benjinyue / hetongjine

df_temp = pd.DataFrame({'总合同金额':hetongjine,
                        '总本金余额':benjinyue,
                        '余额比例':yuebili})
df2 = pd.merge(df2,df_temp,left_on = 'report_id',
               right_index = True)

del df_temp

## 第六部分 scheduled_payment_amount actual_payment_amount

yinghuankuan = df.groupby('report_id')['scheduled_payment_amount'].sum()
shihuankuan = df.groupby('report_id')['actual_payment_amount'].sum()
huankuanbili = shihuankuan / yinghuankuan

df_temp = pd.DataFrame({'应还款':yinghuankuan,
                        '实还款':shihuankuan ,
                        '还款比例':huankuanbili})
df2 = pd.merge(df2,df_temp,left_on = 'report_id',
               right_index = True)

del df_temp

## 第七部分 当前逾期期数 当前逾期金额
yuqiqishu = df.groupby('report_id')['curr_overdue_cyc'].sum()
yuqijine = df.groupby('report_id')['curr_overdue_amount'].sum()
df_temp = pd.DataFrame({'逾期期数':yuqiqishu,
                        '逾期金额':yuqijine})
    
df2 = pd.merge(df2,df_temp,left_on = 'report_id',
               right_index = True)

del df_temp

## 第八部分 还款日期
## 还款提前天数
df['yanhoutianshu'] = pd.to_datetime(df.scheduled_payment_date) - pd.to_datetime(df.recent_pay_date)
df['yanhoutianshu'] = df['yanhoutianshu'].dt.days
yanhoutianshu =  df.groupby('report_id')['yanhoutianshu'].mean()
yanhoutianshu = pd.DataFrame(yanhoutianshu)
yanhoutianshu.columns = ['还款延后天数']

df2 = pd.merge(df2,yanhoutianshu,left_on = 'report_id',
               right_index = True)

del yanhoutianshu
## 发放日期 到期日期不要了

df2.to_csv(eda_path + 'cd_ln_process.csv',index = False)
df2.to_csv(eda_path + 'cd_ln_process2.csv',encoding = 'UTF-8',index = False)
