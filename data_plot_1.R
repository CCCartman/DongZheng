setwd("D:/统计建模")  #设置工作路径
data=read.csv("train.csv",header = T)  #读入数据
attach(data)
library(ggplot2)#绘图
head(data) #显示数据的前几个观测
str(data)#查看变量类型及个数
summary(data)#计算所有变量的基本统计量
salary=as.factor(salary)
y=as.factor(y)
#基本统计
a=table(is_local,y)
a #户籍与y
ggplot(data,aes(x=salary,fill=y))+geom_bar()+coord_polar(theta="x")+scale_fill_brewer(palette = "Accent") #工资与y
ggplot(data,aes(x=y,fill=marry_status))+geom_bar()+facet_wrap(~marry_status,scales="free")#婚姻状况与y
ggplot(data,aes(x=y,fill=edu_level))+geom_bar()+facet_grid(.~edu_level) #学历与y
