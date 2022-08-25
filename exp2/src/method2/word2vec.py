from gensim.models import word2vec
import numpy as np
import json
readpath1 = "../../lab2_dataset/entity_with_text.txt"
readpath2 = "../../lab2_dataset/relation_with_text.txt"
writepath11 = "./output/entity_co.txt"
writepath12 = "./output/entity_vec.txt"
writepath21 = "./output/relation_co.txt"
writepath22 = "./output/relation_vec.txt"
#提取语料
with open(readpath1, 'r') as fr, open(writepath11, 'w') as fw:
    for line in fr.readlines():
        fw.write(line.strip().split('\t')[1])
        fw.write("\n")
with open(readpath2, 'r') as fr, open(writepath21, 'w') as fw:
    for line in fr.readlines():
        fw.write(line.strip().split('\t')[1])
        fw.write("\n")
#训练实体
print("training entity")
sentences = word2vec.LineSentence(writepath11)
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3)
print("over entity")
entity_vec = {}
with open(readpath1, 'r') as fr:
    for line in fr.readlines():
        vec = 0
        id = line.strip().split('\t')[0]
        words = line.strip().split('\t')[1].split(" ")
        for word in words:
            vec += model.wv[word]
        vec = vec / np.linalg.norm(vec)
        entity_vec[id] = vec
#写实体向量
print("writing entity")
with open(writepath12, 'w') as fw:
    for entity in entity_vec.keys():
            fw.write(entity+"\t")
            fw.write(str(entity_vec[entity].tolist()))
            fw.write("\n")
#训练关系
print("training relation")
sentences = word2vec.LineSentence(writepath21)
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3)
print("over relation")
relation_vec = {}
with open(readpath2, 'r') as fr:
    for line in fr.readlines():
        vec = 0
        id = line.strip().split('\t')[0]
        words = line.strip().split('\t')[1].split(" ")
        for word in words:
            vec += model.wv[word]
        vec = vec / np.linalg.norm(vec)
        relation_vec[id]=vec
#写关系向量
print("writing relation")
with open(writepath22, 'w') as fw:
    for relation in relation_vec.keys():
            fw.write(relation+"\t")
            fw.write(str(relation_vec[relation].tolist()))
            fw.write("\n")