from numpy import *
from ast import literal_eval


def distanceL1(t1, t2):
    distvector = t1-t2
    sum = fabs(distvector).sum()
    return sum


def EntityAndRelationOpen(dir, sp="\t"):
    dict = {}
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            Id = line.strip().split(sp)[0]
            vector = array(literal_eval(line.strip().split(sp)[1]))
            dict[Id] = vector
    return dict


def testopen(dir, sp="\t"):
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            head = line.strip().split(sp)[0]
            relation = line.strip().split(sp)[1]
            tail = line.strip().split(sp)[2]
            list.append((head, relation, tail))
    return list


if __name__ == '__main__':
    entitypath = "..//output//entityVector.txt"
    relationpath = "..//output//relationVector.txt"
    testpath = "..//lab2_dataset//test.txt"
    writepath = "..//output//predict.txt"
    entityVector = EntityAndRelationOpen(entitypath)
    relationVector = EntityAndRelationOpen(relationpath)
    trible = testopen(testpath)
    miss = 0
    n = len(trible)
    print("predicting")
    with open(writepath, 'w') as f:
        for k, predictTrible in enumerate(trible):
            h = predictTrible[0]
            r = predictTrible[1]
            if(h in entityVector.keys() and r in relationVector.keys()):
                t_predict = entityVector[h] + relationVector[r]  # 预测向量
                result = {}
                for entity in entityVector.keys():
                    result[entity] = distanceL1(
                        t_predict, entityVector[entity])  # 计算每个向量与预测向量的距离
                order_5 = sorted(result.items(), key=lambda v: v[1], reverse=True)[
                    0:5]  # 根据字典值进行排序
                temp_5 = order_5
            else:
                order_5 = temp_5  # 找不到实体跟随上一次结果
                miss += 1
            for i, entity in enumerate(order_5):
                if i < 4:
                    f.write(entity[0]+',')
                else:
                    f.write(entity[0])
                    if(k != n-1):
                        f.write('\n')
            if(k % 100 == 0):
                print("%d/%d   missnum:%d" % (k, n, miss))
    print("over")
