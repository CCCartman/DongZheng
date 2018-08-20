# “东证期货杯”全国大学生统计建模大赛<br>
=======
# "东证期货杯"统计建模大赛
## 题目
**互联网金融时代下信用评分体系模型的构建**<br>
## 队伍简介
四位成员分别是来自首经贸的芮文豪、姚默涵、伊芯慧、柏元元<br>
## 竞赛简单总结
赛题的目标为基于某贷款机构的历史业务数据运用数据挖掘等方法，构造模型变量，制定信用规则，建立信用评估模型，预测违约情况。<br>

数据集分为:
contest_basic_train基础表-训练集
contest_basic_test基础表-测试集
contest_ext_crd_hd_report机构版征信-报告主表
contest_ext_crd_cd_ln机构版征信-贷款
contest_ext_crd_cd_lnd机构版征信-贷记卡
contest_ext_crd_is_creditcue机构版征信-信用提示
contest_ext_crd_is_sharedebt机构版征信-未销户贷记卡或者未结清贷款
contest_ext_crd_is_ovdsummary机构版征信-逾期(透支)信息汇总
contest_ext_crd_qr_recordsmr机构版征信-查询记录汇总
contest_ext_crd_qr_recorddtlinfo机构版征信-信贷审批查询记录明细
contest_ext_crd_cd_ln_spl机构版征信-贷款特殊交易
contest_ext_crd_cd_lnd_ovd机构版征信-贷记卡逾期/透支记录。<br>

本小组通过对各个表构造统计特征、创建新特征等方式提取各表信息，最后与基础表相连训练XGBoost、LightGBM、RandomForest的加权组合模型。最终在评选中获得全国二等奖(30/1000+)<br>

代码需要改进的地方有很多。从代码写作风格来看，代码的风格太统计化，应该写的更加工程化。从建模角度看，由于时间原因特征工程没有做的更加细致，如应该深挖时间特征构使用滑窗法构造变量，通过特征交叉构造组合特征，可以尝试深度学习或者更多其他方法建模等。

## 文件解释
### 特征工程与数据处理
tsv2csv.py<br>
cd_ln_process.py<br>
cd_lnd_process.py<br>
spl_ovd_process.py<br>
sharedebt_process.py<br>
qihuoout_process.py<br>
ovdsummary_process.py	<br>
read_data.py<br>
process_after_combine.py<br>
make_train_data.py<br>
make_test_data.py<br>
get_both_columns.py	<br>
### 可视化
data_plot_1.R	<br>
data_plot_2.py	<br>
### 建模
train_lgb.py<br>	
train_rf.py<br>
train_xgb.py	<br>
sort_importance.py<br>	
voting.py<br>
predict_test.py<br>
