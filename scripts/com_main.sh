CUDA_VISIBLE_DEVICES=4,5 python /home/dyf/data_generate/doc-instruct/complexity.py \
    --seed_tasks_path /home/dyf/data_generate/doc-instruct/data/lima/epoch/qwen/com_new_instruct_20000_round_3_Qwen2.5-7B-Instruct.jsonl \
    --batch_length 20000 \
    --model_id /data1/dyf/model/Qwen2.5-7B-Instruct \
    --roundi 3