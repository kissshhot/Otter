batch_dir=/home/dyf/data_generate/doc-instruct/data/lima/epoch/qwen/
CUDA_VISIBLE_DEVICES=0,1 python /home/dyf/data_generate/doc-instruct/response_generate.py \
    --seed_tasks_path /home/dyf/data_generate/doc-instruct/data/lima/epoch/qwen/5000.jsonl \
    --model_id /data1/dyf/model/Qwen2.5-7B-Instruct \
    --batch_length 5000 \
    --batch_dir ${batch_dir}
# /home/dyf/data_generate/doc-instruct/data/lima/epoch/diff/filter_words_full_100000.jsonl

# /data1/dyf/model/Llama-3.1-8B-Instruct
# /data1/dyf/model/Llama-3.1-Tulu-3-8B
# /data1/dyf/model/Mistral-7B-Instruct-v0.3