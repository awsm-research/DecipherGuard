import argparse
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from deep_translator import GoogleTranslator
import base64
import random

from attacks.attack_utils import *
from utils.utils import *


def load_and_concatenate_test_csv(file_paths, subset, percentage):
    temp = [file+f"_{percentage}_percent/"+f"{subset}.csv" for file in file_paths]
    # Load each CSV into a list of DataFrames
    dfs = [pd.read_csv(file+f"_{percentage}_percent/"+f"{subset}.csv") for file in file_paths]
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df




# Function to evaluate guardrails
def evaluate_guardrail(model, tokenizer, prompts, categories, dataset_name, attack_name, guardrail_name="LlamaGuard"):
    dataset_results = []
    preds = []
    probs = []

    formatted_prompts = transform_dataset_prompt(prompts)

    for chat in tqdm(formatted_prompts):
        pred = moderate(chat, tokenizer, model)
        prob = moderate_probability(chat, tokenizer, model)

        if 'unsafe' in pred.lower():
            preds.append(1)
        else:
            preds.append(0)
        probs.append(prob[1])


    for idx, prompt in enumerate(prompts):
        result = {
            'prompt': prompt,
            'category': categories[idx],
            'ID': f'{guardrail_name}_{dataset_name}_{attack_name[idx]}_{idx}',
            'attack_method': attack_name[idx],
            'guardrail': guardrail_name,
            'probability': probs[idx],  
            'pred': preds[idx]
        }

        dataset_results.append(result)


    return dataset_results




# Define a function to transform dataset prompts
def transform_dataset_prompt(prompts):
    transformed_data=[]
    for x in prompts:
        transformed_data.append([{"role": "user", "content": x}])
    return transformed_data

# Define the moderation function
def moderate(chat, tokenizer, model, device='cuda'):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
def moderate_probability(chat, tokenizer, model, device='cuda'):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(
        input_ids=input_ids, 
        max_new_tokens=100, 
        pad_token_id=tokenizer.eos_token_id, 
        return_dict_in_generate=True, 
        output_scores=True
    )
    probs = torch.cat(output.scores).softmax(dim=-1)
    return [probs[1, 19193].item(), 1-probs[1, 19193].item()] # [safe,unsafe]

def main():

    # Load data
    all_file_paths = [
                        "./data/split_attack_prompts/AIM_data",
                        "./data/split_attack_prompts/base64_data",
                        "./data/split_attack_prompts/caesar_cipher_data",
                        "./data/split_attack_prompts/CC_data",
                        "./data/split_attack_prompts/combination_data",
                        "./data/split_attack_prompts/DAN_data",
                        "./data/split_attack_prompts/deepInception_data",
                        "./data/split_attack_prompts/dual_use_data",
                        "./data/split_attack_prompts/self_cipher_data",
                        "./data/split_attack_prompts/zulu_data",
                    ] 
    test_df = load_and_concatenate_test_csv(all_file_paths, subset="test", percentage=1) #since all test.csvs are the same, just need to load once
    test_plain = test_df["prompt"].tolist()
    test_plain = list(set(test_plain))
    test_attacked = test_df["transformed_prompt"].tolist()
    categories = test_df['Summarized Category'].tolist()
    methods = test_df['attack_method'].tolist()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    hf_token = ''  #TODO Replace with your token
    model_id = "meta-llama/Llama-Guard-3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, torch_dtype=dtype, device_map=device)

    # transformed_prompts = transform_dataset_prompt(attacked_prompts)

    attacked_results= evaluate_guardrail(model,tokenizer,test_attacked, categories, '80_test', methods, guardrail_name = 'LlamaGuard')
    plain_results= evaluate_guardrail(model,tokenizer,test_plain, categories, '80_test', methods, guardrail_name = 'LlamaGuard')
    # Save results and metrics
    dataset_results = plain_results+attacked_results

    save_results(dataset_results,  '80_test', 'LlamaGuard')
            
 

if __name__ == '__main__':
    main()
