batch_dir=""

# generate new instructions
CUDA_VISIBLE_DEVICES=2,3 python ./ins_generate.py \
    --doc_path ../path_to_doc \
    --is_vllm \
    --model_id Llama-3.1-Tulu-3-8B \
    --batch_length 20000 \
    --batch_dir ${batch_dir}

# complexity enhancement
CUDA_VISIBLE_DEVICES=4,5 python ./complexity.py \
    --seed_tasks_path ../path_to_instructions \
    --model_id Llama-3.1-Tulu-3-8B \
    --batch_length 20000 \
    --roundi 0 \
    --batch_dir ${batch_dir}

# score before next round, The deita library environment needs to be installed separately
python ./deita/data_score_com.py

# complexity enhancement
CUDA_VISIBLE_DEVICES=4,5 python ./complexity.py \
    --seed_tasks_path ../path_to_instructions \
    --model_id Llama-3.1-Tulu-3-8B \
    --batch_length 20000 \
    --roundi 1 \
    --batch_dir ${batch_dir}

# diversity filter
CUDA_VISIBLE_DEVICES=4,5 python ./diversity_filter.py \
    --seed_tasks_path ../path_to_instructions \
    --batch_length 10000 \
    --batch_dir ${batch_dir}

# response generate
CUDA_VISIBLE_DEVICES=0,1 python ./response_generate.py \
    --seed_tasks_path ../path_to_instructions \
    --model_id Llama-3.1-Tulu-3-8B \
    --batch_length 10000 \
    --batch_dir ${batch_dir}

# Convert the data format to share_gpt
# python ./share_gpt.py