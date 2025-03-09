export CUDA_VISIBLE_DEVICES=0

python compress.py \
    --dataset_name asqa \
    --prompt_file prompts/asqa_sum_or_ext.json \
    --eval_file data/asqa_eval_gtr_top100.json \
    --tag snippet \
    --shot 2 \
    --ndoc 5 \
    --seed 42 \
    --model LLM PATH such as meta-llama/Meta-Llama-3-8B-Instruct \
    --temperature 1 \
    --top_p 1 \
    --top_k 1 \
    --max_new_tokens 200
