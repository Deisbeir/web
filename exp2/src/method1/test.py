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
    relationpath = "../output//relationVector.txt"
    testpath = "..//lab2_dataset//dev.txt"
    writepath = "..//output//test.txt"
    entityVector = EntityAndRelationOpen(entitypath)
    relationVector = EntityAndRelationOpen(relationpath)
    trible = testopen(testpath)
    hit1 = 0
    hit5 = 0
    miss = 0
    n = len(trible)
    print("predicting")
    with open(writepath, 'w') as f:
        for k, predictTrible in enumerate(trible):
            h = predictTrible[0]
            r = predictTrible[1]
            t = predictTrible[2]
            if(h in entityVector.keys() and r in relationVector.keys()):
                t_predict = entityVector[h] + relationVector[r]  # 预测向量
                result = {}
                for entity in entityVector.keys():
                    result[entity] = distanceL1(t_predict, entityVector[entity])  # 计算每个向量与预测向量的距离
                order_5 = sorted(result.items(), key=lambda v: v[1], reverse=True)[0:5]  # 根据字典值进行排序
                temp_5 = order_5
            else:
                order_5 = temp_5  # 找不到实体跟随上一次结果
                miss += 1
            if order_5[0][0] == t:
                hit1 += 1
            for i, entity in enumerate(order_5):
                if entity[0] == t:
                    hit5 += 1
                if i < 4:
                    f.write(entity[0]+',')
                else:
                    f.write(entity[0])
                    if(k != n):
                        f.write('\n')
            if(k % 100 == 0):
                print("%d/%d   hit@1:%f  hit@5:%f missnum:%d" %(k, n, hit1/(k+1), hit5/(k+1), miss))
    print("hit@1:%f  hit@5:%f\n" % (hit1/n, hit5/n))
