# 这里会让模型对指令生成多个回答，并采用rejection sampling 和 preference-based sampling来筛选出一个最好的回答
import os
import json
import argparse
from transformers import AutoTokenizer
import vllm
from importlib import import_module
import torch
import copy
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/doc-instruct/data/lima/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/doc-instruct/data/lima/epoch/com/com_new_instruct_round_1.jsonl",
        help="The path to instruction.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None
    )
    parser.add_argument(
        "--batch_length",
        type=int,
        default=10,
        help="response generated",
    )
    return parser.parse_args()

def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    templates.create_prompt_with_huggingface_tokenizer_template
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function

def create_prompt_with_huggingface_tokenizer_template(messages, tokenizer, add_bos=False):
    formatted_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if add_bos:
        formatted_text = tokenizer.bos_token + formatted_text
    return formatted_text

def use_vllm(prompts, model, sampling_params, chat_formatting_function, tokenizer):
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
        formatted_prompts.append(formatted_prompt)
    prompts = formatted_prompts
            
    outputs = model.generate(prompts, sampling_params)
    outputs = [it.outputs[0].text for it in outputs]
    return outputs[0]


def response_generate_main(batch_dir, seed_tasks, model, sampling_params, chat_formatting_function, tokenizer, model_id, batch_length):
    model_id = model_id.split('/')[-1]
    all_logs = []
    copied_list = copy.deepcopy(seed_tasks)
    for t in tqdm(copied_list):
        prompt = t['conversations'][0].strip("*").strip()
        result = use_vllm([prompt], model, sampling_params, chat_formatting_function, tokenizer).strip()
        print(result)
        t['conversations'].append(result)
        all_logs.append(t)
        if len(all_logs) % 500 == 0:
            output_log_jsonl(os.path.join(batch_dir, f"response_{batch_length}_{model_id}.jsonl"), all_logs) 
        if len(all_logs) >= batch_length:
            break
    output_log_jsonl(os.path.join(batch_dir, f"response_{batch_length}_{model_id}.jsonl"), all_logs)

if __name__ == "__main__":
    args = parse_args()
    batch_dir = args.batch_dir
    batch_length = args.batch_length
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    model_id = args.model_id
    chat_formatting_function = dynamic_import_function("templates.create_prompt_with_huggingface_tokenizer_template")
    model = vllm.LLM(
        model=model_id,
        tokenizer=model_id,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        tokenizer_revision=None,
        revision=None,
    )
    
    sampling_params = vllm.SamplingParams(
        temperature=0.0,  # greedy decoding
        max_tokens=5000,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    response_generate_main(batch_dir, seed_tasks, model, sampling_params, chat_formatting_function, tokenizer, model_id, batch_length)
