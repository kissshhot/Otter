from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
from tqdm import tqdm
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tokenizer_embedding = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model_embedding = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5', device_map={"": "cuda"}) # , device_map={"": "cuda"}
model_embedding.eval()

def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_task_path",
        type=str,
        required=True,
        default="",
        help="The path to instructions",
    )
    parser.add_argument(
        "--batch_length",
        type=int,
        default=10,
        help="",
    )
    return parser.parse_args()

def embedding_filter(txt, sentence_embedding):
    encoded_input = tokenizer_embedding(txt, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        model_output = model_embedding(**encoded_input)
        txt_embeddings = model_output[0][:, 0]
    txt_embeddings = torch.nn.functional.normalize(txt_embeddings, p=2, dim=1)
    if sentence_embedding == []:
        sentence_embedding = txt_embeddings
        return True, sentence_embedding
    score_list =[txt_embeddings[0] @ sentence_embedding[i] for i in range(0, len(sentence_embedding))]
    if any(x > 0.9 for x in score_list):
        return False, sentence_embedding
    else:
        sentence_embedding = torch.cat((sentence_embedding, txt_embeddings), dim=0)
        return True, sentence_embedding


if __name__ == "__main__":
    args = parse_args()
    question_embedding = []
    all_logs = []
    # seed_tasks_path = "/home/dyf/data_generate/doc-instruct/data/lima/epoch/com/com_ablation/new_response/com2.jsonl"
    seed_tasks = [json.loads(l) for l in open(args.seed_task_path, "r")]
    batch_size = 5000
    search_batch_size = batch_size
    n_batches = (len(seed_tasks) + search_batch_size - 1) // search_batch_size
    for batch_idx in tqdm(range(n_batches)):
        start_idx = batch_idx * search_batch_size
        end_idx = min((batch_idx + 1) * search_batch_size, len(seed_tasks))
        batch_tasks = seed_tasks[start_idx:end_idx]
        sorted_tasks = sorted(batch_tasks, key=lambda x: x['complexity_score'], reverse=True)
        for task in tqdm(sorted_tasks):
            question = task['conversations'][0]
            f1, tmp = embedding_filter(question, question_embedding)
            if f1:
                question_embedding = tmp
                all_logs.append(task)
                if len(all_logs) >= args.batch_length:
                    break

    output_log_jsonl(os.path.join(args.batch_dir, f"com2_{args.batch_length}_filter_0.9.jsonl"), all_logs)
