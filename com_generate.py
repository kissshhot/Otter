import os
import json
from tqdm import tqdm
from prompts.prompt_template import doc_com_prompt_self
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

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

def otter_com(seed_tasks, roundi, is_vllm, batch_length, model, sampling_params, chat_formatting_function, tokenizer, model_id, batch_dir): #随机选择数据进行generate
    all_logs = []
    model_id = model_id.split('/')[-1]
    if roundi > 0:
        pre_tasks = [json.loads(l) for l in open(f"/home/dyf/data_generate/doc-instruct/data/lima/epoch/mistral/com_new_instruct_{batch_length}_round_{roundi-1}_{model_id}.jsonl", "r")]
    if is_vllm == True:
        for idx in tqdm(range(len(seed_tasks))):
            if roundi > 0:
                if pre_tasks[idx]['complexity_score'] >= seed_tasks[idx]['complexity_score']:
                    all_logs.append(pre_tasks[idx])
                    continue
            question = seed_tasks[idx]['conversations'][0].strip("*").strip()
            prompt = doc_com_prompt_self.format(question=question)
            te = False
            while True:
                result = use_vllm([prompt], model, sampling_params, chat_formatting_function, tokenizer)
                try:
                    if '[New Question]:' in result:
                        question = result.split('[New Question]:')[1].strip("*").strip()
                    break
                except:
                    te = True
                    break
            if te:
                all_logs.append(seed_tasks[idx])
                continue
            seed_tasks[idx]['conversations'][0] = question
            seed_tasks[idx]['result'] = result
            all_logs.append(seed_tasks[idx])
            if len(all_logs) % 500 == 0:
                output_log_jsonl(os.path.join(batch_dir, f"com_new_instruct_{batch_length}_round_{roundi+1}_{model_id}_unscore.jsonl"), all_logs) 
            if len(all_logs) >= batch_length:
                break
    output_log_jsonl(os.path.join(batch_dir, f"com_new_instruct_{batch_length}_round_{roundi+1}_{model_id}_unscore.jsonl"), all_logs)
    return all_logs