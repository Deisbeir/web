import json
import math

read_path = "./output/words_index.json"
output_path = "./output/ID_index.json"
ID_index = {}
with open(read_path, encoding='utf-8') as f:
    datas = json.load(f)
    N = 306242
    j = 0
    for word in datas:
        for i in datas[word]:
            if (i["ID"] not in ID_index):
                j += 1
                ID_index[i["ID"]] = [{
                    "word":
                    word,
                    "value":
                    (1 + math.log10(i["freq"])) * (math.log10(N / len(word)))
                }]
            else:
                ID_index[i["ID"]].append({
                    "word":
                    word,
                    "value":
                    (1 + math.log10(i["freq"])) * (math.log10(N / len(word)))
                })
        print(j)
print("正在写回文件")
with open(output_path, 'w') as f_out:
    json.dump(ID_index, f_out)