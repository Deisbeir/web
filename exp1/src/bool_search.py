import json
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def transfer(input):  # 将输入去停用词，词根化
    stop_words = set(stopwords.words('english'))
    input = PorterStemmer().stem(input)
    if input in stop_words:
        input = " "
    return input


def search(word):  # 搜索该单词对应的文档并返回列表
    search_list = []
    output_path = "./output"
    index=OrderedDict()
    with open(output_path + "/" + "words_index", 'r') as fread:
        index = json.load(fread)
    transed = transfer(word)
    if (transed in index):
        for obj in index[transed]:
            search_list.append(obj["ID"])
    else:
        print("找不到该查询文件")
    return search_list


def calculate(list1, op, list2):  # AND,OR,NOT计算
    if (op == "AND"):
        list3 = list(set(list1).intersection(set(list2)))
    elif (op == "OR"):
        list3 = list(set(list1).union(set(list2)))
    elif (op == "NOT"):
        list3 = list(set(list1).difference(set(list2)))
    return list3


def init(to_search):  # 将非括号与运算符转换为该单词对应结果的列表
    other = ["AND", "OR", "NOT", "(", ")"]
    for i, v in enumerate(to_search):
        if (v not in other):
            to_search[i] = search(v)


def Postfix(to_search):  # 建立后缀表达式
    postfix = []  # 后续表达式栈
    op = []  # 符号栈
    other = ["AND", "OR", "NOT", "(", ")"]
    for i in to_search:
        if (i not in other):
            postfix.append(i)
        elif (i == '('):
            op.append(i)
        elif (i == ')'):
            temp = op.pop(-1)
            while (temp != '('):
                postfix.append(temp)
                temp = op.pop(-1)
        else:
            while (len(op) != 0 and op[-1] != '('):
                postfix.append(op.pop(-1))
            op.append(i)
    while (len(op) != 0):
        postfix.append(op.pop(-1))
    return postfix


def document(postfix):
    ops = ["AND", "OR", "NOT"]
    to_calcu = []
    for w in postfix:
        if (w not in ops):
            to_calcu.append(w)
        else:
            list2 = to_calcu.pop(-1)
            list1 = to_calcu.pop(-1)
            to_calcu.append(calculate(list1, w, list2))
    return to_calcu[0]


input = input("请输入bool查询词：")
to_search = word_tokenize(input)
init(to_search)
print(document(Postfix(to_search)))