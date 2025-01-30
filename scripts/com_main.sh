CUDA_VISIBLE_DEVICES=0,1 python /home/dyf/data_generate/doc-instruct/complexity.py \
    --seed_tasks_path /home/dyf/data_generate/doc-instruct/data/lima/epoch/mistral/com_new_instruct_20000_round_3_Mistral-7B-Instruct-v0.3.jsonl \
    --batch_length 20000 \
    --model_id /data1/dyf/model/Mistral-7B-Instruct-v0.3 \
    --roundi 3