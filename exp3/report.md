# Web实验三: 推荐系统

PB19051033 孙远林 PB19111675 德斯别尔

### SVD++

推荐系统提供给用户关于产品或服务的个性化推荐。这个系统经常依赖于协同过滤(CF),即根据过去的交易进行分析建立用户和物品间的联系。CF的两种比较成功的方法分别是隐因子模型和邻域模型，前者直接描述用户和产品，后者分析产品或用户之间的相似性。通过使用SVD++模型隐因子模型和邻域模型现在可以顺利合并，从而建立更准确的组合模型。通过利用用户的显式和隐式反馈扩展模型，进一步提高了推荐准确性。

##### 理论

###### 基线估计

典型的CF数据显示出大的用户和项目效果-例如：即一些用户给出的评分比其他用户高，以及某些商品获得的评分比其他商品高的系统性倾向。通常考虑到这些影响来调整数据，我们将这些影响封装在基线估算中。用μ表示总体的平均评分。未知评分$r_{ui}$的基线估计由$b_{ui}$表示，并说明用户和项目的影响:
$b_{ui} = \mu + b_u + b_i $  (1)
参数 $b_u$ 和 $b_i$ 分别表示用户u和项目i与平均值的观察偏差。
为了估计 $b_u$ 和 $b_i$ ，一个最小二乘问题可以解决:
$\min_{b^*} \sum_{(u,i)\in K}(r_{ui} - \mu - b_u -bi ) + \lambda_1 (\sum_u b_u^2 + \sum_i b_i^2)$ 
这里，第一项尝试去算出 [公式] 和 [公式] 符合给定的评分。第二项正则化项通过惩罚参数的大小来避免过拟合。

###### 领域模型

最常见的CF方法是基于邻域模型。它是基于用户的。这种面向用户的方法基于对有相同想法的用户的记录评分来估计未知评分。使用类似的基于物品的方法。在这种方法中，评分是根据同一用户对类似项目的已知评分来估计的。更好的可伸缩性和改进的准确性使面向项目的方法在许多情况下更有利。并且基于物品的方法更适合解释预测背后的原因。因为用户熟悉他们以前喜欢的东西，却不认识那些据称有兴趣的用户。因此，我们尝试了是基于物品的方法。

###### 隐因子模型

隐因子模型包含了协同过滤的一种替代方法，其更全面的目标是发现解释所观察评分的隐含特征；比如 pLSA，神经网络，潜在狄利克雷分布模型。我们将使用用户-物品评分矩阵上的奇异值分解(SVD)所诱导的模型。SVD模型由于其准确性和可伸缩性应该具有较好的效果(网上许多教程也是这么推荐到）。一个典型的模型将每个用户u与用户因子向量 $p_u \in R^f$ 关联起来,并且每个项目i与一个项目因子向量 $q_i \in R^f$ 关联起来。预测是通过取内积来实现的，即 $ \hat r_{ui} = b_{ui} + \sum_{j \in S^k_{(i;u)}} \theta^u_{ij}(r_{uj}-n_{uj})$ 。比较复杂的部分是参数估计。

在信息检索中，利用奇异值分解(SVD)来识别潜在的语义因子是很好的方法。然而，在CF领域中应用SVD会遇到困难，因为缺失评分的比例很高。此外，如果只对相对较少的已知条目进行处理，就很容易出现过拟合。早期的工作依靠估算来填充缺失评分并且使得评分矩阵变得稠密。然而，估算可能会非常昂贵，因为它会显著增加数据量。此外，由于不准确的估算，数据可能会有很大的失真。因此，建模直接只观察评分，而避免过拟合通过适当的正则化模型，如:
$ \min_{p_*,q_*,b_*} \sum_{(u,i) \in K}(r_{ui} - \mu - b_u - b_i - p^T_uq_i)^2 + \lambda_3(||p_u||^2 + ||q_i||^2 + b^2_u + b^2_i)$ (5)

采用简单的梯度下降法求解(5)。

Paterek提出了相关的NSVD模型，该模型避免显式地参数化每个用户，而是基于用户所评分的物品对用户进行建模。这样，每个项目i都与两个因子向量 $q_i$ 和 $x_i$ 相关。用户u的表示是通过求和实现的: $(\sum_{j \in R(u)} x_i ) / \sqrt {R(u)} $ ,所以 $r_{ui}$ 由下式预测： $b_{ui}+q^T_i(\sum_{j \in R(u)} x_i ) / \sqrt {R(u)} $。这里，R(u)是由用户u评分过的项目集合。

##### SVD++模型

在SVD中，在预测用户i对物品j的评分时，只考虑了用户特征向量和商品特征向量，以及评分相对于平均分的偏置向量。在SVD++中，更进一步的多考虑了用户对其所有有过评分行为的商品的隐式反馈。
$p(U_i,M_j)= \mu + b_i + b_u + M^T_j(U_i + \frac{1}{\sqrt{|N_i|^2}*\sum_{j \in N_i}}y_i)$
其中$N_i$表示用户i的行为记录(包括浏览和评论过的商品集合)。$y_i$为隐藏的评价了物品j的个人喜好偏置。式中除以$\sqrt{|N_i|^2}$的目的是为了抑制用户评分数量对模型的影响。
$y_j=y_j+\eta \{[\sum_{j \in M, i \in U}(p(U_i,M_j)-V_{ij})*q_i \frac{1}{\sqrt{|N_i|^2}}] - \lambda * y_j\} $

##### 代码说明

###### 数据存储格式：
```
    with open(trainpath, "r") as fr:
        for i, line in enumerate(fr.readlines()):
            scores = line.split('\t')[1].split(' ')
            for j, score in enumerate(scores):
                item = int(score.split(',')[0])
                grade = int(score.split(',')[1])
                if grade != -1:
                    train_data.append([i,item,grade])
```
将数据存储为 [[user_id,item_id,rating],[user_id,item_id,rating],[user_id,item_id,rating]...]

###### 建立用户对物品的行为的集合
```
        self.u_dict={}
        for i in range(self.mat.shape[0]):
            uid=self.mat[i,0]
            iid=self.mat[i,1]
            self.u_dict.setdefault(uid,[])
            self.u_dict[uid].append(iid)
            self.bi.setdefault(iid,0)
            self.bu.setdefault(uid,0)
            self.qi.setdefault(iid,np.random.random((self.K,1))/10*np.sqrt(self.K))
            self.pu.setdefault(uid,np.random.random((self.K,1))/10*np.sqrt(self.K))
            self.y.setdefault(iid,np.zeros((self.K,1))+.1)
```
使用一个字典存储用户对物品的行为集合，键为用户 id，值为该用户有过行为的商品列表。遍历所有训练数据得到。

###### 隐式反馈
```
    #计算sqrt_Nu和∑yj
    def getY(self,uid,iid):
        Nu=self.u_dict[uid]
        I_Nu=len(Nu)
        sqrt_Nu=np.sqrt(I_Nu)
        y_u=np.zeros((self.K,1))
        if I_Nu==0:
            u_impl_prf=y_u
        else:
            for i in Nu:
                y_u+=self.y[i]
            u_impl_prf = y_u / sqrt_Nu
        
        return u_impl_prf,sqrt_Nu
```
使用函数来计算用户多有过评分商品集合的隐式反馈，并在后续的模型训练中，乘以学习率，逐步对其进行调整。

###### 评分预测
```
    def predict(self,uid,iid):  #预测评分的函数
        #setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu及用户评价过的物品u_dict，并设置初始值为0
        self.bi.setdefault(iid,0)
        self.bu.setdefault(uid,0)
        self.qi.setdefault(iid,np.zeros((self.K,1)))
        self.pu.setdefault(uid,np.zeros((self.K,1)))
        self.y.setdefault(uid,np.zeros((self.K,1)))
        self.u_dict.setdefault(uid,[])
        u_impl_prf,sqrt_Nu=self.getY(uid, iid)
        rating=self.avg+self.bi[iid]+self.bu[uid]+np.sum(self.qi[iid]*(self.pu[uid]+u_impl_prf)) #预测评分公式
        #由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        if rating>5:
            rating=5
        if rating<1:
            rating=1
        return rating
```
根据公式直接对评分进行计算，并在后续模型训练时，根据隐式反馈逐步对其进行调整，是其更接近最好的情况。

###### 训练模型
```
    def train(self,steps=30,gamma=0.04,Lambda=0.15):    #训练函数，step为迭代次数。
        print('train data size',self.mat.shape)
        for step in range(steps):
            print('step',step+1,'is running')
            KK=np.random.permutation(self.mat.shape[0]) #随机梯度下降算法，kk为对矩阵进行随机洗牌
            rmse=0.0
            if step == 0:
                for i in range(self.mat.shape[0]):
                    self.result.update({self.mat[KK[i],0]:{}})
            for i in range(self.mat.shape[0]):
                j=KK[i]
                uid=self.mat[j,0]
                iid=self.mat[j,1]
                rating=self.mat[j,2]
                predict=self.predict(uid, iid)
                self.result[uid].update({iid:predict})
                u_impl_prf,sqrt_Nu=self.getY(uid, iid)
                eui=rating-predict
                rmse+=eui**2
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])  
                self.bi[iid]+=gamma*(eui-Lambda*self.bi[iid])
                self.pu[uid]+=gamma*(eui*self.qi[iid]-Lambda*self.pu[uid])
                self.qi[iid]+=gamma*(eui*(self.pu[uid]+u_impl_prf)-Lambda*self.qi[iid])
                for j in self.u_dict[uid]:
                    self.y[j]+=gamma*(eui*self.qi[j]/sqrt_Nu-Lambda*self.y[j])
                                    
            gamma=0.93*gamma
            print('rmse is',np.sqrt(rmse/self.mat.shape[0]))
```
对于每一项参数乘以学习率，通过模型训练来计算隐式反馈，通过梯度下降来逐步接近最好的情况。
##### 结果


##### 遇到问题
模型训练时间过长，导致无法具体调整参数，以及修复bug需要很长的时间，导致结果不太如意。