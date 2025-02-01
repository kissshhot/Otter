from deita.selection.scorer import Llama_Scorer
import json
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_name_or_path = "/data1/dyf/model/deita-complexity-scorer/" # hkust-nlp/deita-complexity-scorer
scorer = Llama_Scorer(model_name_or_path, is_vllm = True)

def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

tasks_path = '/home/dyf/data_generate/doc-instruct/data/lima/epoch/self_few_shot/diff_new_instruct_12000_self_few_shot_round_0_Llama-3.1-Tulu-3-8B.jsonl'
tasks = [json.loads(l) for l in open(tasks_path, "r")]
for idx in tqdm(range(len(tasks))):
    input_text = tasks[idx]['conversations'][0].strip()
    complexity_score = scorer.infer_complexity(input_text)
    tasks[idx]['complexity_score'] = complexity_score
    print(input_text)
output_log_jsonl('/home/dyf/data_generate/doc-instruct/data/lima/epoch/self_few_shot/com0_self_few_shot.jsonl', tasks)