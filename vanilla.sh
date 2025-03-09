export CUDA_VISIBLE_DEVICES=0

python vanilla.py \
    --dataset_name asqa \
    --prompt_file prompts/asqa_vanilla.json \
    --eval_file data/asqa_eval_gtr_top100.json \
    --tag vanilla \
    --shot 2 \
    --ndoc 5 \
    --seed 42 \
    --model LLM PATH such as meta-llama/Meta-Llama-3-8B-Instruct \
    --temperature 1 \
    --top_p 1 \
    --top_k 1 \
    --max_new_tokens 200
