import json

def convert_jsonl_to_sharegpt(input_file, output_file):
    sharegpt_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            
            conversation_data = {
                "id": f"conversation_{i + 1}",
                "conversations": []
            }
            conversations = data.get("conversations", [])
            if len(conversations) == 1:
                continue
            if conversations:
                for j, text in enumerate(conversations):
                    if j > 1:
                        print(i)
                    role = "user" if j % 2 == 0 else "assistant"
                    conversation_data["conversations"].append({
                        "role": role,
                        "content": text.strip()
                    })
            sharegpt_data.append(conversation_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

input_file = ""
output_file = ""
convert_jsonl_to_sharegpt(input_file, output_file)