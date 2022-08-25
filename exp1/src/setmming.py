import os
import json
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

    
# 获取文件路径
def get_file_path():
    read_path = "./dataset/US_Financial_News_Articles"
    output_path = "./output"
    return read_path,output_path
# 读取文件名称和内容
def get_files():
    # 获取文件输入和输出路径
    read_path, output_path = get_file_path()
    folder_names = os.listdir(read_path)
    # 获取read_path下的所有文件名称（顺序读取的）
    i = 0
    inverted_index = {}
    for folder_name in folder_names:
        files = os.listdir(read_path + "/" + folder_name)
        for file_name in files:
            # 读取单个文件内容
            with open(read_path + "/" + folder_name + "/" + file_name,encoding='utf-8') as f_obj:
                datas = json.load(f_obj)
            # 处理单个文件(调用方法)
            finish_datas = get_deal_file(
                datas["text"] + datas["title"],
                folder_name + "/" + file_name,
            )
            # 输出结果到指定路径下
            if (i == 0):
                inverted_index = finish_datas
            else:
                for j in finish_datas:
                    if (j in inverted_index):
                        inverted_index[j].append(finish_datas[j][0])
                    else:
                        inverted_index[j] = finish_datas[j]
            i = i + 1
            print(i)
    key_index = sorted(inverted_index.items(), key=lambda x: len(x[1]))
    words_index = OrderedDict()
    for i in range(len(key_index)):
        words_index[key_index[i][0]] = key_index[i][1]
    with open(output_path + "/" + "words_index.json", 'w') as f_obj:
        json.dump(words_index, f_obj)
    print("文件处理完毕")


# 处理单个文件程序 /针对不同批量处理文件进行修改对文件的处理代码,返回值：finish_dfdata/
def get_deal_file(data_text, file_name):
    our_str = "zxcvbnmasdfghjklqwertyuiopZXCVBNMASDFGHJKLQWERTYUIOP"
    for s in data_text:
        if (s not in our_str):
            data_text = data_text.replace(s, " ")
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(data_text)
    preprocessed_sent = {}
    for w in word_tokens:
        ww = PorterStemmer().stem(w)  # 词根化
        if ww not in stop_words:  # 去停用词
            if (ww in preprocessed_sent):
                preprocessed_sent[ww][0][
                    "freq"] = preprocessed_sent[ww][0]["freq"] + 1
            else:
                preprocessed_sent[ww] = [{
                    "ID": file_name,
                    "freq": 1,
                }]
    return preprocessed_sent


# 主函数
if __name__=="__main__":

    # 开始处理文件，并输出处理文件结果
    get_files()
