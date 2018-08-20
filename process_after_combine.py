# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:56:09 2018

@author: rwhca
"""
import numpy as np
import pandas as pd
from read_data import readData

def groupImmpute(df,col,switch = 'mean',group = False):
    # val 0 val 1
    if group:
        if switch == 'mean':
            val = df.groupby('y')[col].mean().tolist()
            
        if switch == 'median':
            val = df.groupby('y')[col].median().tolist()
        
        if switch == 'mode':
            val = df.groupby('y')[col].mode().tolist()
        
        
        idx0 = df[df[col].isnull() & (df.y == 0)].index
        idx1 = df[df[col].isnull() & (df.y == 1)].index
        
        print(val)
        df.loc[idx0,col] = val[0]
        df.loc[idx1,col] = val[1]
        
    else:
        val = df.loc[:,col].mean()
        
        idx = df[df[col].isnull()].index
        df.loc[idx,col]  = val
        
    return df[col]

def processData(filePath,file_1,train = True):
    
    df = readData(filePath,file_1)
       
    print("================这是一条分割线=============")
    
    
    df = df.drop(['unnamed: 0'],axis = 1)
    ## 先连接其他数据
    df.drop([
             '未结清贷款account_count',
             '未销户贷记卡account_count',
             '未结清贷款credit_limit',
             '未销户贷记卡credit_limit',
             '未销户贷记卡max_credit_limit_per_org',
             '未销户贷记卡min_credit_limit_per_org',
             '未结清贷款balance',
             '未销户贷记卡used_credit_limit',
             '未结清贷款latest_6m_used_avg_amount',
             '未销户贷记卡latest_6m_used_avg_amount',
             'changing_amount',
             'last_months'],axis = 1,inplace = True)
    if train:    
        df_temp = pd.read_csv('D:\\workspace python\\contest\\qihuoout.csv')
    else:
        df_temp = pd.read_csv('D:\\workspace python\\contest\\out_test.csv')
    df = pd.merge(df,df_temp,left_on = 'report_id',
                      right_on = 'report_id',how = 'left')
    del df_temp
    
    #df.sort_values(by='y',inplace = True)
    ## 变量名与变量格式修改
    df.drop('id_card',axis = 1,inplace = True)
    
    df.drop('loan_date',axis = 1,inplace = True)
    
    df.drop('agent',axis = 1,inplace = True)
    
	
    ## 以后再处理了
    df.work_province.isnull().sum()
    
    ## edu
    df.edu_level.value_counts()
    mapping = {'本科':'本科及以上',
               '硕士研究生':'本科及以上',
               '博士研究生':'本科及以上',
               '硕士及以上':'本科及以上',
               '高中':'专科以下',
               '初中':'专科以下',
               '专科及以下':'专科以下',
               '其他':'专科以下',
               '专科':'专科'}
    
    df.edu_level = df.edu_level.map(mapping)
    df.edu_level = df.edu_level.fillna('missing')
    
    del mapping
    ## 公积金
    df.has_fund = df.has_fund.fillna('missing')
    
    ## 婚姻
    df.marry_status.value_counts()
    mapping = {'离婚':'离婚',
               '离异':'离婚',
               '丧偶':'其他',
               '已婚':'已婚',
               '未婚':'未婚',
               '其他':'其他'}
    df.marry_status = df.marry_status.map(mapping)
    
    del mapping
    
    ## 收入
    df.salary.isnull().sum()
    
    
    df.salary = df.salary.fillna('missing')
    ## y
    if train:
        df.y = df.y.astype('category')
    
    '''
    pd.crosstab(df.y,df.salary).div(
            pd.crosstab(
                    df.y,df.salary).sum(1).astype(float),axis = 0).plot(
                    kind = 'bar')
    '''      
    ####------下一个
    df['ln_settle_rate'] = df.贷款结清比例
    del df['贷款结清比例']
    
    df['ln_settle_rate'] = groupImmpute(df,'ln_settle_rate')
    '''
    df.boxplot('ln_settle_rate',by = 'y')
    '''
    ## 贷款是否异常
    df['ln_abnormal'] = df.贷款是否异常
    del df['贷款是否异常']
    
	
	df.ln_abnormal = df.ln_abnormal.fillna('missing')
    '''
    pd.crosstab(df.y,df.ln_abnormal).div(
            pd.crosstab(
                    df.y,df.ln_abnormal).sum(1).astype(float),axis = 0).plot(
                    kind = 'bar')
    '''
    
    ## 贷款免担保比例
    df['ln_guarantee_rate'] = df.贷款免担保比例
    del df['贷款免担保比例']     
    
    df['ln_guarantee_rate'].isnull().sum()
    '''
    df.boxplot('ln_guarantee_rate',by = 'y')
    '''
    df['ln_guarantee_rate'] = groupImmpute(df,'ln_guarantee_rate')
    
    
    ## 贷款按周期归还
    df['ln_payment_month'] = df.贷款按周期归还
    del df['贷款按周期归还']
    '''
    df.boxplot('ln_payment_month',by = 'y')
    '''
    df['ln_payment_month'] = groupImmpute(df,'ln_payment_month')
    
    ## 贷款不定期归还
    df['ln_payment_random'] = df.贷款不定期归还
    del df['贷款不定期归还']
    
    '''
    pd.crosstab(df.y,df.ln_payment_random).div(
            pd.crosstab(
                    df.y,df.ln_payment_random).sum(1).astype(float),axis = 0).plot(
                    kind = 'bar')
     ''' 
    
	df.ln_payment_random = df.ln_payment_random.fillna('missing')
    ## 贷款一次性归还
    
    df['ln_payment_ontime'] = df.贷款一次性归还
    del df['贷款一次性归还']
    
    '''
    pd.crosstab(df.y,df.ln_payment_ontime).div(
            pd.crosstab(
                    df.y,df.ln_payment_ontime).sum(1).astype(float),axis = 0).plot(
                    kind = 'bar')
    '''
    
    df.ln_payment_ontime= df.ln_payment_ontime.fillna('missing')

    
    ## 贷款按其他方式归还
    df['ln_payment_other'] = df.贷款按其他方式归还
    del df['贷款按其他方式归还']
    '''
    pd.crosstab(df.y,df.ln_payment_other).div(
            pd.crosstab(
                    df.y,df.ln_payment_other).sum(1).astype(float),axis = 0).plot(
                    kind = 'bar')
            '''
    del df['ln_payment_other']
    
    ## 贷款逾期程度
    df['ln_overdue'] = df.贷款逾期程度
    del df['贷款逾期程度']
    
    df['ln_overdue'] = groupImmpute(df,'ln_overdue')
    
    ## 总合同金额
    df['ln_conamount'] = df.总合同金额
    del df['总合同金额']
         
    df['ln_conamount'] = groupImmpute(df,'ln_conamount')
    
    ## 总本金余额
    df['ln_prin_balance'] = df.总本金余额
    del df['总本金余额']
    
    #df.boxplot('ln_prin_balance',by = 'y')
    
    df['ln_prin_balance'] = groupImmpute(df,'ln_prin_balance')
    
    ## 余额比例
    df['ln_balance_rate'] = df.余额比例
    del df['余额比例']
    
    df['ln_balance_rate'] = groupImmpute(df,'ln_balance_rate')
    
    #df.boxplot('ln_balance_rate',by = 'y')
    
    ## 实还款
    df['ln_actrepayment'] = df.实还款
    del df['实还款']
    
    df['ln_actrepayment'] = groupImmpute(df,'ln_actrepayment')
    
    ## 应还款
    df['ln_should_repayment'] = df.应还款
    del df['应还款']
    
    df['ln_should_repayment'] = groupImmpute(df,'ln_should_repayment')
    
    ## 还款比例
    df['ln_repayment_rate'] = df.还款比例
    del df['还款比例']
    
    df['ln_repayment_rate'] = groupImmpute(df,'ln_should_repayment')
    
    ## 逾期期数
    df['ln_overdue_month'] = df.逾期期数
    del df['逾期期数']
    
    df['ln_overdue_month'] = groupImmpute(df,'ln_overdue_month')
    
    ## 逾期金额
    df['ln_overdue_amount'] = df.逾期金额
    del df['逾期金额']
    
    df['ln_overdue_amount'] = groupImmpute(df,'ln_overdue_amount')
    
    ## 还款延后天数
    df['ln_overdue_days'] = df.还款延后天数
    del df['还款延后天数']
    
    df['ln_overdue_days'] = groupImmpute(df,'ln_overdue_days')
    
    df.drop(['first_loan_open_month','first_loancard_open_month',
             'first_sl_open_month'],axis = 1,inplace = True)
    
    for col in ['house_loan_count','commercial_loan_count',
              'other_loan_count',
              'loancard_count',
              'standard_loancard_count',
              'announce_count','dissent_count']:
        df.loc[:,col] = groupImmpute(df,col) 
    
    
    ## 贷款审查次数
    df['query_times'] = df.贷款审查次数
    del df['贷款审查次数']
    
    df['query_times'] = groupImmpute(df,'query_times')
    
    ## 担保人审查次数
    df['security_query_times'] = df.担保人审查次数
    del df['担保人审查次数']
    
    df['security_query_times'] = groupImmpute(df,'security_query_times')
    
    ## card一类的数据处理
    card_nameList = df.columns.tolist()
    card_nameList = [col for col in card_nameList
                     if 'card' in col[:4]]
    
    
    for col in card_nameList:
        df.loc[df[col] == np.inf,col] = 0
        df[col] = groupImmpute(df,col)
        
    name_list_temp = ['准贷记卡60天以上透支count_dw',
             '贷款逾期count_dw',
             '贷记卡逾期count_dw',
             '准贷记卡60天以上透支months',
             '贷款逾期months',
             '贷记卡逾期months',
             '准贷记卡60天以上透支highest_oa_per_mon',
             '贷款逾期highest_oa_per_mon',
             '贷记卡逾期highest_oa_per_mon',
             '准贷记卡60天以上透支max_duration',
             '贷款逾期max_duration',
             '贷记卡逾期max_duration']
    
    name_list_temp_2 = [i.replace('准贷记卡60天以上透支','semi_card_').replace('贷款逾期','loan_').replace(
            '贷记卡逾期','card') for i in name_list_temp]
    
    
    df[name_list_temp_2] = df.loc[:,name_list_temp]
    for i in name_list_temp:
        del df[i]
    
    del name_list_temp
    
    for name_ in df[name_list_temp_2]:
        df[name_] = groupImmpute(df,name_,'mean')
    
    del name_list_temp_2
    
    df.card_unlogout_account_count = groupImmpute(df,'card_unlogout_account_count')
    
    ## 信用卡审批次数
    del df['信用卡审批次数']
    
    ## 删除省份
    del df['work_province']
    return df

if __name__ == '__main__':
    df = processData(filePath = 'D:\\workspace python\\contest\\EDA\\trainDocument\\',file_1 = 'contest_basic_train.csv')
    print(df.columns)
    print(df.isnull().sum())