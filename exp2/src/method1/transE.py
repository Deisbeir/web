from random import uniform, sample
from numpy import *
from copy import deepcopy


class TransE:
    def __init__(self, entityList, relationList, tripleList, margin=1, lr=1, dim=100, L1=True):
        self.margin = margin
        self.lr = lr  # 学习率
        self.dim = dim  # 向量维度
        self.entityList = entityList  # 为实体列表，初始化后为一个字典，key是entity，values是其narray向量
        self.relationList = relationList  # 同上
        self.tripleList = tripleList  # 同上
        self.loss = 0  # 损失值
        self.L1 = L1  # 范数

    def initialize(self):
        # 初始化向量
        entityVectorList = {}
        relationVectorList = {}
        for entity in self.entityList:  # 初始化实体向量
            n = 0
            entityVector = []
            while n < self.dim:
                ram = init(self.dim)  # 初始化的范围
                entityVector.append(ram)
                n += 1
            entityVector = norm(entityVector)  # 归一化
            entityVectorList[entity] = entityVector
        print("实体向量初始化完成，共有%d" % len(entityVectorList))
        for relation in self. relationList:  # 初始化关系向量
            n = 0
            relationVector = []
            while n < self.dim:
                ram = init(self.dim)  # 初始化的范围
                relationVector.append(ram)
                n += 1
            relationVector = norm(relationVector)  # 归一化
            relationVectorList[relation] = relationVector
        print("关系向量初始化完成，共有%d" % len(relationVectorList))
        self.entityList = entityVectorList
        self.relationList = relationVectorList

    def transE(self, cycle=1000):
        bench_size = 128
        print("traning")
        for cycleId in range(cycle):
            Sbatch = sample(self.tripleList, bench_size)  # 选出的样本
            Tbatch = []  # 三元组对 ：{((h,r,t),(h',r,t'))}
            for sbatch in Sbatch:  # 对样本三元组构造三元组队
                TriplePair = (sbatch, self.TriplePair(sbatch))
                if(TriplePair not in Tbatch):
                    # 排除三元组中不在entitylist中的实体
                    if(sbatch[0] in self.entityList.keys() and sbatch[2] in self.entityList.keys()):
                        Tbatch.append(TriplePair)
            self.update(Tbatch)  # 更新向量表
            if cycleId % 100 == 0:
                print("第%d次循环" % cycleId)
                print("本100次循环loss总值为%f" % self.loss)
                self.loss = 0
                self.lr = self.lr*0.98  # 学习率逐步下降

    def TriplePair(self, triplet):
        # 获取随机替换头实体或尾实体的三元组
        i = uniform(-1, 1)
        if i < 0:  # 小于0，替换头实体
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]
                if entityTemp != triplet[0]:
                    break
            corruptedTriplet = (entityTemp, triplet[1], triplet[2])
        else:  # 大于等于0，替换尾实体
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]
                if entityTemp != triplet[2]:
                    break
            corruptedTriplet = (triplet[0], triplet[1], entityTemp)
        return corruptedTriplet

    def update(self, Tbatch):  # 更新向量表
        copyEntityList = deepcopy(self.entityList)
        copyRelationList = deepcopy(self.relationList)

        for TriplePair in Tbatch:
            # 获取Tbatch更新后的实体和关系向量
            headVector = copyEntityList[TriplePair[0][0]]
            tailVector = copyEntityList[TriplePair[0][2]]
            relationVector = copyRelationList[TriplePair[0][1]]
            correpted_head = copyEntityList[TriplePair[1][0]]
            correpted_tail = copyEntityList[TriplePair[1][2]]

            # 获取Tbatch更新前的实体和关系向量
            headVector0 = self.entityList[TriplePair[0][0]]
            tailVector0 = self.entityList[TriplePair[0][2]]
            relationVector0 = self.relationList[TriplePair[0][1]]
            correpted_head0 = self.entityList[TriplePair[1][0]]
            correpted_tail0 = self.entityList[TriplePair[1][2]]
            # 根据范数计算距离
            if self.L1:
                distTriplet = distanceL1(
                    headVector0, relationVector0, tailVector0)  # 正样本
                distCorruptedTriplet = distanceL1(
                    correpted_head0, relationVector0, correpted_tail0)  # 负样本
            else:
                distTriplet = distanceL2(
                    headVector0, relationVector0, tailVector0)
                distCorruptedTriplet = distanceL2(
                    correpted_head0, relationVector0, correpted_tail0)
            eg = self.margin + distTriplet - distCorruptedTriplet
            if eg > 0:
                self.loss += eg  # 计算损失量，这里为100个Tbatch的累加
                if self.L1:
                    # 梯度下降更新
                    tempPositive = 2 * \
                        (tailVector0 - headVector0 - relationVector0)  # 计算梯度
                    tempNegtative = 2 * \
                        (correpted_tail0 - correpted_head0 - relationVector0)
                    for i in range(len(tempPositive)):
                        if tempPositive[i] >= 0:
                            tempPositive[i] = 1
                        else:
                            tempPositive[i] = -1
                        if tempNegtative[i] >= 0:
                            tempNegtative[i] = 1
                        else:
                            tempNegtative[i] = -1
                else:
                    tempPositive = 2 * self.lr * \
                        (tailVector0 - headVector0 - relationVector0)
                    tempNegtative = 2 * self.lr * \
                        (correpted_tail0 - correpted_head0 - relationVector0)
                # 更新实体和关系向量
                headVector += self.lr * tempPositive
                tailVector -= self.lr * tempPositive
                relationVector += self.lr * (tempPositive - tempNegtative)
                # 替换了tail时，基于负样本更新
                if(TriplePair[0][0] == TriplePair[1][0]):
                    headVector -= self.lr * tempNegtative
                    correpted_tail += self.lr * tempNegtative
                # 替换了head时，基于负样本更新
                elif(TriplePair[0][2] == TriplePair[1][2]):
                    correpted_head -= self.lr * tempNegtative
                    tailVector += self.lr * tempNegtative
                # 归一化刚刚刚更新的向量
                copyEntityList[TriplePair[0][0]] = norm(headVector)
                copyEntityList[TriplePair[0][2]] = norm(tailVector)
                copyRelationList[TriplePair[0][1]] = norm(relationVector)
                copyEntityList[TriplePair[1][0]] = norm(correpted_head)
                copyEntityList[TriplePair[1][2]] = norm(correpted_tail)
        # 最终Tbatch后更新回原字典
        self.entityList = copyEntityList
        self.relationList = copyRelationList

    def writeEntilyVector(self, dir):  # 写实体向量
        print("写入实体")
        entityVectorFile = open(dir, 'w')
        for entity in self.entityList.keys():
            entityVectorFile.write(entity+"\t")
            entityVectorFile.write(str(self.entityList[entity].tolist()))
            entityVectorFile.write("\n")
        entityVectorFile.close()

    def writeRelationVector(self, dir):  # 写关系向量
        print("写入关系")
        relationVectorFile = open(dir, 'w')
        for relation in self.relationList.keys():
            relationVectorFile.write(relation + "\t")
            relationVectorFile.write(str(self.relationList[relation].tolist()))
            relationVectorFile.write("\n")
        relationVectorFile.close()


def init(dim):  # 返回随机值
    return uniform(-6/(dim**0.5), 6/(dim**0.5))


def distanceL1(h, r, t):
    # L1距离
    s = h + r - t
    sum = fabs(s).sum()
    return sum


def distanceL2(h, r, t):
    # L2距离
    s = h + r - t
    sum = (s*s).sum()
    return sum


def norm(list):
    # 归一化
    var = linalg.norm(list)
    i = 0
    while i < len(list):
        list[i] = list[i]/var
        i += 1
    return array(list)


def openDetailsAndId(dir, sp="\t"):
    # 获取实体列表
    idNum = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(DetailsAndId[0])
            idNum += 1
    return idNum, list


def openTrain(dir, sp="\t"):
    # 获取训练三元组表
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if(len(triple) < 3):
                continue
            list.append(tuple(triple))
            num += 1
    return num, list


if __name__ == '__main__':
    dirEntity = "..//lab2_dataset//entity_with_text.txt"
    entityIdNum, entityList = openDetailsAndId(dirEntity)
    dirRelation = "..//lab2_dataset//relation_with_text.txt"
    relationIdNum, relationList = openDetailsAndId(dirRelation)
    dirTrain = "..//lab2_dataset//train.txt"
    tripleNum, tripleList = openTrain(dirTrain)
    print("打开TransE")
    transE = TransE(entityList, relationList, tripleList, margin=1, dim=75)
    print("TranE初始化")
    transE.initialize()
    transE.transE(3000)
    transE.writeRelationVector("..//output//relationVector.txt")
    transE.writeEntilyVector("..//output//entityVector.txt")
