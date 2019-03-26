# DSTH
DSTH190327  

1. 将数据data.hy与hash标签hashcode.hy放入"DSTH\datasets\cifar10\"中

2. Hash函数训练  
	cd HashFuncLearning  
	python dsthMain.py --train True  
  
3.加载DSTH Model获取Hash码  
	cd HashCodePrediction  
	python getHashCodes.py
