batch_dir=/home/dyf/data_generate/doc-instruct/data/lima/epoch/com/com_ablation/
CUDA_VISIBLE_DEVICES=6,7 python /home/dyf/data_generate/doc-instruct/continue_response.py \
    --seed_tasks_path /home/dyf/data_generate/doc-instruct/data/lima/epoch/com/com_ablation/com2_last5000.jsonl \
    --model_id /data1/dyf/model/Llama-3.1-Tulu-3-8B \
    --batch_length 5000 \
    --batch_dir ${batch_dir}


# /data1/dyf/model/Llama-3.1-8B-Instruct
# /data1/dyf/model/Llama-3.1-Tulu-3-8B
# /data1/dyf/model/Mistral-7B-Instruct-v0.3