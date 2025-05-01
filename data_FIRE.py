import csv
import json

def process_fire_file(file_path):
    # 打开CSV文件
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        # 创建CSV阅读器
        csv_reader = csv.reader(csvfile)
        json_array = []

        # 逐行读取并处理
        for row in csv_reader:
            type = row[0]

            if type != 'ClassicalChinese':
                continue
            input = row[1]
            output = row[2]

            item = {
                "instruction": "翻译下面的文言文和现代文，如果是文言文，则翻译成现代文，如果是现代文，则翻译成文言文",
                "input": input,
                "output": output,
                "system": "你是一名精通古汉语和文言文的专家"
            }

            json_array.append(item)
    print(len(json_array))
    return json_array

