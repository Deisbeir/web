import json
import math
import operator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

read_path1 = "./output/words_index.json"
read_path2 = "./output/ID_index.json"
text = input("请输入语义查询的内容：")
our_str = "zxcvbnmasdfghjklqwertyuiopZXCVBNMASDFGHJKLQWERTYUIOP"
for s in text:
    if (s not in our_str):
        text = text.replace(s, " ")
words = word_tokenize(text)
stopWords = set(stopwords.words('english'))
wordsFiltered = []
for w in words:
    if w not in stopWords:
        wordsFiltered.append(PorterStemmer().stem(w))
print(wordsFiltered)
search = {}
for word in wordsFiltered:
    if (word in search):
        search[word] += 1
    else:
        search[word] = 1
with open(read_path1, encoding='utf-8') as f1:
    datas = json.load(f1)
N = 58000
search_len = 0
for w in search:
    if (w in datas):
        search[w] = t = (1 + math.log10(search[w])) * (math.log10(
            N / len(datas[w])))
        search_len += t * t
    else:
        search[w] = 0
f1.close()
search_len = math.sqrt(search_len)
with open(read_path2, encoding='utf-8') as f2:
    datas = json.load(f2)
rank = []
for ID in datas:
    ID_len = 0
    mult = 0
    for ID_word in datas[ID]:
        ID_len += ID_word["value"] * ID_word["value"]
        if (ID_word["word"] in search):
            mult += ID_word["value"] * search[ID_word["word"]]
    ID_len = math.sqrt(ID_len)
    rank.append({"ID": ID, "tf_idf": mult / (ID_len * search_len)})

rank = sorted(rank, key=operator.itemgetter('tf_idf'), reverse=True)
for i in range(10):
    print(rank[i]["ID"])

