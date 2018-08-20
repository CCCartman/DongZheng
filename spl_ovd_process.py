#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 15:29:35 2018

@author: apple
"""

import pandas as pd 
df_spl=pd.read_csv("/Users/apple/Documents/master/统计建模/contest_csv/contest_ext_crd_cd_ln_spl.csv")
def ln_spl(df):
    jine_sum=df['changing_amount'].groupby(df['report_id']).sum()
    return jine_sum

df_spl=pd.DataFrame(ln_spl(df_spl))
df_spl['report_id'] = df_spl.index.values.reshape(-1,1)

df_ovd=pd.read_csv("/Users/apple/Documents/master/统计建模/contest_csv/contest_ext_crd_cd_lnd_ovd.csv",encoding="gbk")
last_months=pd.DataFrame(df_ovd['LAST_MONTHS'].groupby(df_ovd['REPORT_ID']).sum())
amount=pd.DataFrame(df_ovd['AMOUNT'].groupby(df_ovd['REPORT_ID']).sum())
df_spl=pd.merge(df_spl,last_months,left_index=True,right_index=True)
df_spl.to_csv("/Users/apple/Documents/master/统计建模/contest_csv/spl&ovd.csv",encoding="utf-8",index='False')
    
    