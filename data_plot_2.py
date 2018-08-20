# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:10:30 2018

@author: asus
"""

#导入相应的库
import pandas as pd #数据处理
#绘图
import seaborn as sns
#显示出图形
%matplotlib inline 
import matplotlib as mpl
import matplotlib.pyplot as plt
train=pd.read_csv('train.csv') #读入数据
train.head() #查看数据前几个观测
train.info() #查看数据信息
train.describe()#描述统计

#用每个id的平均本金金额除以每个id的平均合同金额
sns.boxplot(x="y", y="ln_balance_rate", hue="ln_abnormal",data=train, palette="Set3")  

#将每个id的未销户最近6个月平均使用额度求和
sns.violinplot(x="y", y="card_unlogout_latest_6m_used_avg_amount", hue="ln_abnormal", data=train)

#住房公积金对房贷的影响
sns.stripplot(x="y", y="house_loan_count", hue="has_fund",data=train, jitter=True,palette="Set2", split=True)

#按准贷记卡分类的贷款逾期笔数  （均值）
plt.subplot(131)
sns.barplot(x="y", y="semi_card_count_dw", hue="ln_abnormal", data=train,ci=0)
plt.subplot(132)
#按贷记卡分类的贷款逾期笔数
sns.barplot(x="y", y="cardcount_dw", hue="ln_abnormal", data=train,ci=0)
plt.subplot(133)
