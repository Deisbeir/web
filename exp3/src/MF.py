from math import pow
import numpy as np


def MF(R, P, Q, N, M, K, steps=5000, alpha=0.0002, beta=0.02):
    # 矩阵因子分解函数，
    # steps：迭代次数，alpha：步长
    for step in range(steps):
        # 进行迭代
        for i in range(N):
            for j in R[i].keys():
                eij = R[i][j] - np.dot(P[i, :], Q[:, j])  # .dot表矩阵乘
                for k in range(K):
                    # 更新参数
                    if R[i][j] != -1:  # 限制评分不为-1，即预测评分
                        P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        e = 0  # 误差
        # 求损失函数：
        for i in range(N):
            for j in R[i].keys():
                if R[i][j] != -1:
                    e += pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    # 加入正则化
                    for k in range(K):
                        e += (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        print(step, e)
    return P, Q


if __name__ == "__main__":
    readpath = "..//data//train.txt"  # 训练集
    writepath = "..//submit//best.txt"
    R = []
    N = M = 0
    with open(readpath, "r") as fr:
        for i, line in enumerate(fr.readlines()):
            scores = line.split('\t')[1].split(' ')
            for j, score in enumerate(scores):
                item = int(score.split(',')[0])
                grade = int(score.split(',')[1])
                if j == 0:
                    R.append({item: grade})
                else:
                    R[i][item] = grade
                if M < item + 1:
                    M = item + 1
    N = i + 1
    print(N, M)
    K = 10  # k值可根据需求改变
    P = np.random.rand(N, K)
    Q = np.random.rand(K, M)
    nP, nQ = MF(R, P, Q, N, M, K)
    with open(writepath, 'w') as fw:
        for i in range(N):
            temp = {}
            for j in range(M):
                temp[j] = np.dot(P[i, :], Q[:, j])
            RANK100 = sorted(temp.items(), key=lambda v: v[1],
                             reverse=True)[0:100]  # 根据字典值进行排序
            fw.write(str(i) + '\t')
            for j, recommend in enumerate(RANK100):
                if j < 99:
                    fw.write(str(recommend[0]) + ',')
                else:
                    fw.write(str(recommend[0]))
                    if (i != N - 1):
                        fw.write('\n')
            if i % 100 == 0:
                print(i)
