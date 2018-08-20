# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:48:09 2018

@author: rwhca
"""

import numpy as np
import pandas as pd
import os

def set_os(filePath):
    os.chdir(filePath)

def seeData():
    for file in os.listdir():
        print(file)
        df_temp = pd.read_csv(file,encoding='utf-8')
        print(df_temp.shape)
        print('***********************')
        del df_temp    
    

def readData(filePath,file_1):
    set_os(filePath)
    df = pd.read_csv(filePath + file_1,encoding = 'utf-8')
    df.columns = df.columns.str.lower()
    fileList = os.listdir()
    fileList.remove(file_1)
    
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