import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import os
import json

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams


class LLModel:
    def __init__(self, model):
        self.model = LLM(
            model=model, 
            enable_prefix_caching=True,
            trust_remote_code=True,
            # max_model_len=8192
        )

    def generate(self, prompt, sampling_params):

        text = self.model.get_tokenizer().apply_chat_template(
            prompt, 
            add_generation_prompt=True,
            tokenize=False
        )
        output = self.model.generate(text, sampling_params)[0].outputs[0].text

        return output

def make_doc_prompt(doc, doc_id, doc_prompt):
    text = doc['text']
    return doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))

def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, test=False):
    message = []
    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "") 
        else:
            doc_list = item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)

    message.append({
        "role": "user",
        "content": prompt.replace("{A}", "").rstrip()
    })
    if not test:
        message.append({
            "role": "assistant",
            "content": "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        })
    return message

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="asqa/eli5/hotpotqa/musique")
    parser.add_argument("--prompt_file", type=str, help="prompt template file in prompts folder")
    parser.add_argument("--eval_file", type=str, help="eval file in data folder")
    parser.add_argument("--tag", type=str, help="vericite/vanilla/snippet/summary/apo")

    parser.add_argument("--shot", type=int, help="number of examples given to LLM")
    parser.add_argument("--ndoc", type=int, help="top-k passages retrieved")
    parser.add_argument("--seed", type=int, help="random seed")

    parser.add_argument("--model", type=str, help="LLM")

    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--max_new_tokens", type=int)

    args = parser.parse_args()

    for _ in args.__dict__:
        print(f"{_}: {args.__dict__[_]}")

    llm = LLModel(args.model)
    stop = []
    if "llama-3" in args.model.lower():
        stop=["<|eot_id|>"]
    elif "glm-4" in args.model.lower():
        stop=["<|user|>"]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        stop=stop
    )
    
    args.max_length = llm.model.llm_engine.model_config.max_model_len
    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")

    np.random.seed(args.seed)

    prompt_data = json.load(open(args.prompt_file))
    eval_data = json.load(open(args.eval_file))

    if "llama-3" in args.model.lower() or "qwen" in args.model.lower():
        few_shot = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    else:
        few_shot = []

    train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)
    for train_id in train_ids:
        train_item = prompt_data["demos"][train_id]
        ndoc = args.ndoc
        few_shot.extend(
            make_demo(
                train_item, 
                prompt=prompt_data["demo_prompt"], 
                ndoc=ndoc, 
                doc_prompt=prompt_data["doc_prompt"], 
                instruction=prompt_data["instruction"]
            )
        )

    logger.info("Generating prompts...") 
    incomplete_doc_list = 0 # For some questions there might be fewer than ndoc documents
    for idx, eval_item in enumerate(tqdm(eval_data)):
        eval_data[idx]['prompt'] = few_shot + make_demo(
            eval_item, 
            prompt=prompt_data["demo_prompt"], 
            ndoc=args.ndoc, 
            doc_prompt=prompt_data["doc_prompt"],
            instruction=prompt_data["instruction"], 
            test=True
        )
        doc_list = eval_item["docs"][:args.ndoc]
        eval_data[idx]['docs'] = doc_list
        if len(doc_list) < args.ndoc:
            incomplete_doc_list += 1
    logger.info("Done.")
    if incomplete_doc_list > 0:
        logger.warning(f"There are {incomplete_doc_list} questions that have incomplete document list (may due to a lot of them are filtered out by summary/extraction).")

    for idx, item in enumerate(tqdm(eval_data)):
        prompt = item['prompt']

        if idx == 0:
            print(prompt)

        output = llm.generate(prompt, sampling_params)
        
        item['prompt'] = prompt
        item['output'] = output

        logger.info(f"Question: {item['question']}")
        logger.info(f"Final model output: {item['output']}") 
        
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"{args.dataset_name}-{model_name}-{args.tag}-shot{args.shot}-ndoc{args.ndoc}-{args.seed}"

    eval_data = {
        "args": args.__dict__,
        "data": eval_data,
    }
    if not os.path.exists("result"):
        os.makedirs("result")
    json.dump(eval_data, open("result/" + name + ".json", "w"), indent=4)

if __name__ == "__main__":
    main()
