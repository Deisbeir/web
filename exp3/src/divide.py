readpath = "..//data//DoubanMusic.txt"
trainpath = "..//data//train.txt"  # 训练集
devpath = "..//data//dev.txt"  # 验证集,每个用户的倒数第二个评分
testpath = "..//data//test.txt"  # 测试集,每个用户的最后一个评分

with open(readpath, 'r') as fr, open(trainpath, 'w') as ftrain, open(
        devpath, 'w') as fdev, open(testpath, 'w') as ftest:
    for line in fr.readlines():
        id = line.strip().split('\t')[0]
        words = line.strip().split('\t')[1:]
        ftrain.write(id + "\t" + " ".join(words[:-3]) + "\n")
        fdev.write(id + "\t" + words[-2] + "\n")
        ftest.write(id + "\t" + words[-2] + "\n")
