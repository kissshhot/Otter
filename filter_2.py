from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import pdb
import torch
import numpy as np
import os
import json
import vllm
from importlib import import_module
import random
import re
import string
from tqdm import tqdm
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tokenizer_embedding = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model_embedding = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5', device_map={"": "cuda"}) # , device_map={"": "cuda"}
model_embedding.eval()
# todo: 加入embedding过滤和质量过滤方法

def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def embedding_filter(txt, sentence_embedding):
    # Tokenize sentences
    encoded_input = tokenizer_embedding(txt, padding=True, truncation=True, return_tensors='pt').to('cuda')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model_embedding(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        txt_embeddings = model_output[0][:, 0]
    # normalize embeddings
    txt_embeddings = torch.nn.functional.normalize(txt_embeddings, p=2, dim=1)
    if sentence_embedding == []:
        sentence_embedding = txt_embeddings
        return True, sentence_embedding
    score_list =[txt_embeddings[0] @ sentence_embedding[i] for i in range(0, len(sentence_embedding))]
    # sentence_embedding = torch.cat((sentence_embedding, txt_embeddings), dim=0)
    if any(x > 0.9 for x in score_list):
        print('embedding不符')
        return False, sentence_embedding
    else:
        print('embedding符合要求')
        sentence_embedding = torch.cat((sentence_embedding, txt_embeddings), dim=0)
        return True, sentence_embedding


def doc_filter(txt, doc):
    # Tokenize sentences
    encoded_input = tokenizer_embedding([txt,doc], padding=True, truncation=True, return_tensors='pt').to('cuda')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model_embedding(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    score = sentence_embeddings[0] @ sentence_embeddings[1]
    if score > 0.7:
        return True
    else:
        return False

if __name__ == "__main__":
    question_embedding = []
    all_logs = []
    seed_tasks_path = "/home/dyf/data_generate/doc-instruct/data/lima/epoch/com/com_ablation/new_response/com2.jsonl"
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    batch_size = 5000
    # distance_threshold = args.distance_distance_threshold
    # search_space_size = args.search_space_size
    search_batch_size = batch_size
    n_batches = (len(seed_tasks) + search_batch_size - 1) // search_batch_size
    for batch_idx in tqdm(range(n_batches)):
        start_idx = batch_idx * search_batch_size
        end_idx = min((batch_idx + 1) * search_batch_size, len(seed_tasks))
        batch_tasks = seed_tasks[start_idx:end_idx]
        sorted_tasks = sorted(batch_tasks, key=lambda x: x['complexity_score'], reverse=True)
        total_com = 0
        # for com in seed_tasks:
        #     total_com += com['complexity_score']
        # avg_com = total_com / len(seed_tasks)
        for task in tqdm(sorted_tasks):
            question = task['conversations'][0]
            # doc = task['doc']
            # if doc_filter(question, doc):
            #     continue
            f1, tmp = embedding_filter(question, question_embedding)
            if f1 :# or task['complexity_score'] >= avg_com: # filter_output(documents, question) and filter_output(questioner_doc, questioner) and f1 and f2: # and filter_output(respondent_doc, respondent): # and quality_score_vllm(question, model, sampling_params, chat_formatting_function):
                # _, question_embedding = embedding_filter(question, question_embedding)
                question_embedding = tmp
                # documents.append(question)
                # questioner_doc.append(questioner)
                # respondent_doc.append(respondent)
                print(question)
                all_logs.append(task)
                if len(all_logs) >= 20000:
                    # output_log_jsonl(os.path.join('/home/dyf/data_generate/doc-instruct/data/lima/epoch/filter_2/', f"10000_filter_2.jsonl"), all_logs)
                    break

    output_log_jsonl(os.path.join('/home/dyf/data_generate/doc-instruct/data/lima/epoch/com/com_ablation/new_response/diverse_filter/', f"com2_10000_filter_0.9.jsonl"), all_logs)
