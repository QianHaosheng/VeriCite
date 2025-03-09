import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import json
import os
import re

import numpy as np
import torch
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

def make_evidence_demo(item, prompt, instruction=None, doc_id=0, test=False):
    message = []
    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question']).replace("{D}", item["docs"][doc_id]["text"])
    message.append({
        "role": "user",
        "content": prompt.replace("{A}", "").rstrip()
    })
    if not test:
        message.append({
            "role": "assistant",
            "content": item["answer"]
        })
    return message

def make_initial_demo(item, prompt, doc_prompt, instruction=None, test=False):
    message = []
    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    if "{D}" in prompt:
        text = "".join([make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(item["docs"])])
        prompt = prompt.replace("{D}", text)

    message.append({
        "role": "user",
        "content": prompt.replace("{A}", "").rstrip()
    })
    if not test:
        message.append({
            "role": "assistant",
            "content": item["answer"]
        })
    return message

def make_summary_demo(item, prompt, doc_prompt, instruction, test=False):
    message = []
    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    if "{D}" in prompt:
        text = "".join([make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(item["docs"])])
        prompt = prompt.replace("{D}", text)
    statements = []
    for statement in item["answer_statements"]:
        text = statement["statement"]
        for doc_id in statement["doc_id"]:
            text += "[" + str(doc_id+1) + "]"
        statements.append(text)
    statements = "\n".join(statements)
    prompt = prompt.replace("{S}", statements)
    message.append({
        "role": "user",
        "content": prompt.replace("{A}", "").rstrip()
    })
    if not test:
        message.append({
            "role": "assistant",
            "content": item["answer"]
        })
    return message

def make_messages(model_name):
    if "llama-3" in model_name.lower() or "qwen" in model_name.lower():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    else:
        messages = []
    return messages

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

def filter_evidence_answer(answer_filter, nli_model, doc_text, id, evidence_answer):

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(nli_model, torch_dtype=torch.bfloat16).to("cuda:1")
        autoais_tokenizer = AutoTokenizer.from_pretrained(nli_model, use_fast=False)

    sents = sent_tokenize(evidence_answer)
    for sent in sents:
        input_text = "premise: {} hypothesis: {}".format(doc_text, sent)
        input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
        with torch.inference_mode():
            outputs = autoais_model.generate(input_ids, max_new_tokens=10)
        result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if result == "1":
            answer_filter.append({
                "statement": sent,
                "doc_id": [id]
            })

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def filter_initial_answer(answer_filter, nli_model, docs, initial_answer):

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(nli_model, torch_dtype=torch.bfloat16).to("cuda:1")
        autoais_tokenizer = AutoTokenizer.from_pretrained(nli_model, use_fast=False)

    sents = sent_tokenize(initial_answer)
    sents_without_citation = [remove_citations(sent).strip() for sent in sents]
    for sent_id, sent in enumerate(sents):
        ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)]
        ref = list(set(ref))
        if len(ref) == 0 or any([ref_id >= len(docs) for ref_id in ref]):
            continue
        else:
            doc_text = "\n".join([docs[psgs_id]["text"] for psgs_id in ref])
            input_text = "premise: {} hypothesis: {}".format(doc_text, sents_without_citation[sent_id])
            input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
            with torch.inference_mode():
                outputs = autoais_model.generate(input_ids, max_new_tokens=10)
            result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if result == "1":
                answer_filter.append({
                    "statement": sent,
                    "doc_id": ref
                })

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
    parser.add_argument("--nli", type=str, help="nli model")

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

    for idx, eval_item in enumerate(tqdm(eval_data)):

        eval_item["docs"] = eval_item["docs"][:args.ndoc]

        ### Initial Answer Generation
        initial_answer_filter = []
        initial_answer_few_shot = make_messages(args.model)
        initial_answer_few_shot_ids = np.random.choice(len(prompt_data["initial_answer_demos"]), args.shot, replace=False)
        for initial_answer_few_shot_id in initial_answer_few_shot_ids:
            initial_answer_few_shot_item = prompt_data["initial_answer_demos"][initial_answer_few_shot_id]
            initial_answer_few_shot.extend(make_initial_demo(
                initial_answer_few_shot_item,
                prompt_data["initial_answer_demo_prompt"],
                prompt_data["initial_answer_doc_prompt"],
                prompt_data["initial_answer_instruction"],
            ))
        eval_item["initial_answer_few_shot"] = initial_answer_few_shot
        
        initial_answer_prompt = make_initial_demo(
            eval_item,
            prompt_data["initial_answer_demo_prompt"],
            prompt_data["initial_answer_doc_prompt"],
            prompt_data["initial_answer_instruction"],
            test=True
        )
        initial_answer = llm.generate(initial_answer_few_shot+initial_answer_prompt, sampling_params)
        eval_item["initial_answer_prompt"] = initial_answer_prompt
        eval_item["initial_answer"] = initial_answer
        filter_initial_answer(
            initial_answer_filter, 
            args.nli, 
            eval_item["docs"], 
            initial_answer
        )

        ### Supporting Evidence Selection
        ### check
        evidence_check_few_shot = make_messages(args.model)
        evidence_check_few_shot_ids = np.random.choice(len(prompt_data["evidence_check_demos"]), args.shot, replace=False)
        for evidence_check_few_shot_id in evidence_check_few_shot_ids:
            evidence_check_few_shot_item = prompt_data["evidence_check_demos"][evidence_check_few_shot_id]
            evidence_check_few_shot.extend(make_evidence_demo(
                evidence_check_few_shot_item,
                prompt_data["evidence_check_demo_prompt"],
                prompt_data["evidence_check_instruction"]
            ))
        eval_item["evidence_check_few_shot"] = evidence_check_few_shot

        evidence_check_prompt = []
        evidence_check = []
        for doc_id in range(min(args.ndoc, len(eval_item["docs"]))):
            evidence_check_prompt.append(make_evidence_demo(
                eval_item,
                prompt_data["evidence_check_demo_prompt"],
                prompt_data["evidence_check_instruction"],
                doc_id=doc_id,
                test=True
            ))
            output = llm.generate(evidence_check_few_shot+evidence_check_prompt[-1], sampling_params)
            evidence_check.append(output)

        eval_item["evidence_check_prompt"] = evidence_check_prompt
        eval_item["evidence_check"] = evidence_check

        ### answer
        evidence_answer_few_shot = make_messages(args.model)
        evidence_answer_few_shot_ids = np.random.choice(len(prompt_data["evidence_answer_demos"]), args.shot, replace=False)
        for evidence_answer_few_shot_id in evidence_answer_few_shot_ids:
            evidence_answer_few_shot_item = prompt_data["evidence_answer_demos"][evidence_answer_few_shot_id]
            evidence_answer_few_shot.extend(make_evidence_demo(
                evidence_answer_few_shot_item,
                prompt_data["evidence_answer_demo_prompt"],
                prompt_data["evidence_answer_instruction"]
            ))
        eval_item["evidence_answer_few_shot"] = evidence_answer_few_shot

        evidence_answer_prompt = []
        evidence_answer = []
        evidence_filter = []

        for doc_id in range(min(args.ndoc, len(eval_item["docs"]))):
            evidence_answer_prompt.append(make_evidence_demo(
                eval_item,
                prompt_data["evidence_answer_demo_prompt"],
                prompt_data["evidence_answer_instruction"],
                doc_id=doc_id,
                test=True
            ))
            if eval_item["evidence_check"][doc_id].startswith("No"):
                evidence_answer.append("")
            else:
                output = llm.generate(evidence_answer_few_shot+evidence_answer_prompt[-1], sampling_params)
                evidence_answer.append(output)
                filter_evidence_answer(
                    evidence_filter, 
                    args.nli, 
                    eval_item["docs"][doc_id]["text"], 
                    doc_id,
                    evidence_answer[-1]
                )
        eval_item["evidence_answer_prompt"] = evidence_answer_prompt
        eval_item["evidence_answer"] = evidence_answer

        eval_item["answer_statements"] = evidence_filter + initial_answer_filter

        ### Final Answer Refinement
        summary_few_shot = make_messages(args.model)
        summary_few_shot_ids = np.random.choice(len(prompt_data["summary_demos"]), args.shot, replace=False)
        for summary_few_shot_id in summary_few_shot_ids:
            summary_few_shot_item = prompt_data["summary_demos"][summary_few_shot_id]
            summary_few_shot.extend(make_summary_demo(
                summary_few_shot_item,
                prompt_data["summary_demo_prompt"],
                prompt_data["summary_doc_prompt"],
                prompt_data["summary_instruction"],
            ))
        eval_item["summary_few_shot"] = summary_few_shot

        summary_prompt = make_summary_demo(
            eval_item,
            prompt_data["summary_demo_prompt"],
            prompt_data["summary_doc_prompt"],
            prompt_data["summary_instruction"],
            test=True
        )
        summary = llm.generate(summary_few_shot+summary_prompt, sampling_params)
        eval_item["summary_prompt"] = summary_prompt
        eval_item["output"] = summary

        logger.info(f"Question: {eval_item['question']}")
        logger.info(f"Final model output: {eval_item['output']}") 
        
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
