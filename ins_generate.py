import argparse
import json
import os
from diff_generate import otter_generate
import vllm
from importlib import import_module
import torch
from transformers import AutoTokenizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="/home/dyf/data_generate/doc-instruct/data/lima/response/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--doc_path",
        type=str,
        required=True,
        default="/home/dyf/data_generate/doc-instruct/data/falcon.jsonl",
        help="The path to the documents.",
    )
    parser.add_argument(
        "--is_vllm",
        action="store_true",
    )
    parser.add_argument(
        "--batch_length",
        type=int,
        default=10,
        help="ins generated each batch",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="/data1/dyf/model/Llama-3.1-Tulu-3-8B",
        help="The path to the model",
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


if __name__ == "__main__":
    args = parse_args()
    all_logs = []
    batch_dir = args.batch_dir
    model_id = args.model_id
    documents = [json.loads(l) for l in open(args.doc_path, "r")]
    if args.is_vllm == True:
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
    otter_generate(documents, args.batch_length, args.is_vllm, model, sampling_params, chat_formatting_function, tokenizer, model_id, batch_dir)