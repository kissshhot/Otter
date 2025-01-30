import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/data1/dyf/model/Llama-3-8B/", use_fast='store_true') # ,use_fast='store_true'
i = 0
data = []

import json


def filter_and_save_jsonl(input_file_path, output_file_path, max_length):
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in tqdm(infile):
            data = json.loads(line)
            # 检查数据长度是否小于100
            text = data['conversations'][0] + data['conversations'][1] # data['messages'][0]['content'] + data['messages'][1]['content']
            input_ids = tokenizer.encode(text)
            if len(input_ids) <= max_length:
                outfile.write(json.dumps(data) + "\n")

# linguistician mathematician programmer musician
# 使用示例
input_file_path = "/home/dyf/data_generate/doc-instruct/data/lima/epoch/com/com_ablation/new_response/q6w.jsonl"  # 输入文件路径
output_file_path = "/home/dyf/data_generate/doc-instruct/data/lima/epoch/com/com_ablation/new_response/q6w_length.jsonl"  # 输出文件路径
max_length = 2048  # 设置最大长度为100
filter_and_save_jsonl(input_file_path, output_file_path, max_length)