import os
import json


def transform_json_content(contents):
    transformed_contents = []

    for content in contents:
        transformed_item = {
            "instruction": "将下面的文言文翻译成现代文",
            "input": content["source"],
            "output": content["target"],
            "system": "你是一名精通古汉语和文言文的专家"
        }
        reversed_transformed_item = {
            "instruction": "将下面的现代文翻译成文言文",
            "input": content["target"],
            "output": content["source"],
            "system": "你是一名精通古汉语和文言文的专家"
        }

        transformed_contents.append(transformed_item)
        transformed_contents.append(reversed_transformed_item)

    return transformed_contents


def read_and_transform_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        original_data_list = json.load(file)  # 加载整个 JSON 列表

        transformed_data = []  # 存储所有转换后的数据

        # 循环遍历列表中的每一个元素，并提取 contents 进行转换
        for original_data_entry in original_data_list:
            contents = original_data_entry.get("contents", [])

            transformed_contents = transform_json_content(contents)

            transformed_data.extend(transformed_contents)  # 将转换后的数据加入到最终结果中

    return transformed_data


def process_directory(directory):
    """ 遍历指定目录及其子目录中的 part1 和 part2 JSON 文件 """
    global transformed_data
    for root, dirs, files in os.walk(directory):

        transformed_data = []
        if 'part1.json' in files:
            part1_path = os.path.join(root, 'part1.json')
            print(part1_path)
            try:
                transformed_data.extend(read_and_transform_file(part1_path))
            except Exception as e:
                print(f'Error reading files: {e}')

        if 'part2.json' in files:
            part2_path = os.path.join(root, 'part2.json')
            print(part2_path)
            try:
                transformed_data.extend(read_and_transform_file(part2_path))
            except Exception as e:
                print(f'Error reading files: {e}')
    print(len(transformed_data))
    return transformed_data
