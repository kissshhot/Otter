import os
import json
import re
from tqdm import tqdm
import sys
from prompts.prompt_template import doc_attr_prompt_self
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

def otter_generate(documents, batch_length, is_vllm, model, sampling_params, chat_formatting_function, tokenizer, model_id, batch_dir):
    model_id = model_id.split('/')[-1]
    all_logs = []
    if is_vllm == True:
        for idx in tqdm(range(0, len(documents))):
            doc = documents[idx]['doc']
            prompt = doc_attr_prompt_self.format(doc=doc)
            et = False
            while True:
                try:
                    result = use_vllm([prompt], model, sampling_params, chat_formatting_function, tokenizer)
                except:
                    et = True
                    break
                try:
                    if '### New Questions:' in result:
                        text_question = result.split('### New Questions:')[1].strip('"').strip()
                        attributions = result.split('### Attributes:')[1].split('### New Questions:')[0].strip('"').strip()
                    break
                except:
                    et = True
                    break
            if et:
                continue
            print(prompt)
            print(result)
            try:
                pattern = r'\d+\.\s*\*\*.*?\*\*:\s*(.*?)(?=\n\d+|\Z)'
                questions = re.findall(pattern, text_question, re.DOTALL)
                if len(questions) == 0:
                    questions = text_question.split('\n')
                    questions = [q.split('. ', 1)[1] for q in questions if q[0].isdigit()]
            except:
                continue
            for question in questions:
                t = {}
                t['doc'] = doc
                t['attributions'] = attributions
                t['conversations'] = []
                t['conversations'].append(question)
                if True:
                    print(question)
                    all_logs.append(t)
                    if len(all_logs) % 500 == 0:
                        output_log_jsonl(os.path.join(batch_dir, f"com_new_instruct_{batch_length}_round_0_{model_id}_unscore.jsonl"), all_logs)
                if len(all_logs) >= batch_length:
                    output_log_jsonl(os.path.join(batch_dir, f"com_new_instruct_{batch_length}_round_0_{model_id}_unscore.jsonl"), all_logs)
                    sys.exit(0)
    return all_logs