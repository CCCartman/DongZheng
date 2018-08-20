# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:27:56 2018

@author: rwhca
"""
### --------------------读取相关模块--------------------
import numpy as np
import pandas as pd
import os

### --------------------模块读取完毕--------------------


### --------------------读取数据并处理--------------------
### 工作路径
file_path = 'd:\\workspace python\\contest\\'

### 把tsv数据集转换为csv数据集
def tsv2csv(filePath,outputPath):
    fileList = [file for file in os.listdir(filePath) if '.tsv' in file]
    for fileName in fileList:
        temp_file = pd.read_table(filePath + fileName,sep='\t')
        temp_file.to_csv(outputPath + fileName.split('.')[0] + '.csv',index = False,encoding = 'utf-8')
        del temp_file
        
tsv2csv(file_path,outputPath = file_path + 'new_csvfile\\')    

### 此时所有tsv数据都已经变成csv数据


