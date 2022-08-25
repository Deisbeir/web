代码运行环境：
	 python3 以及 gensim 库（from gensim.models import word2vec）

代码组织结构：
	在exp2/之下增加了lab2_dataset文件夹，即为实验数据文件夹
	在src/下包含method1，method2两个文件夹，其中method1为使用transE进行预测分析，method为transE+word2ve进行分析
	在method文件夹内除了源代码之外，还包括output文件夹，其中内容为储存训练的临时文件。

编译运行方式：
	在method1中，
		编译运行transE.py文件	->	对原数据进行训练
		编译运行test.py文件		->	用于在线下测试一部分数据集，预估进行预测的hit率
		编译运行predict.py文件	->	生成所需要提交的预测文件
	在method2中，
		编译运行word2vec.py文件	->	对每一个实体或关系生成初始化的向量数据，用于transE计算
		其他文件与method1文件相同

关键函数为TransE中的TransE()函数以及update()函数，实现细节可见函数注释。
		