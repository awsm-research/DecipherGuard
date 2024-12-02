import argparse
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from deep_translator import GoogleTranslator
import base64
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from attacks.attack_utils import *
import evaluate
from utils.utils import *


def calculate_perplexity(prompts, perplexity):
    results = perplexity.compute(model_id='gpt2',
                             add_start_token=False,
                             predictions=prompts)
    
    return results['perplexities']


def load_and_concatenate_test_csv(file_paths, subset, percentage):
    temp = [file+f"_{percentage}_percent/"+f"{subset}.csv" for file in file_paths]
    # Load each CSV into a list of DataFrames
    dfs = [pd.read_csv(file+f"_{percentage}_percent/"+f"{subset}.csv") for file in file_paths]
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


# Function to evaluate guardrails
def evaluate_guardrail(perplexity, prompts, categories, dataset_name, attack_name, guardrail_name="Perplexity"):

    preds = []
    dataset_results = []
    perplexities = calculate_perplexity(prompts, perplexity)
    for p in perplexities:
        if p > 106.9:#arbitary threshold
            preds.append(1)
        else:
            preds.append(0)
        
    
    for idx, prompt in enumerate(prompts):
        result = {
            'prompt': prompt,
            'category': categories[idx],
            'ID': f'{guardrail_name}_{dataset_name}_{attack_name[idx]}_{idx}',
            'attack_method': attack_name[idx],
            'guardrail': guardrail_name,
            'probability': perplexities[idx],
            'pred': preds[idx]
        }
        dataset_results.append(result)

    return dataset_results




def main():
    # Load datasets
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

    perplexity = evaluate.load("perplexity", module_type="metric")

    all_results = []

    # transformed_prompts = transform_dataset_prompt(attacked_prompts)
    attacked_results= evaluate_guardrail(perplexity,test_attacked, categories, '80_test', methods, guardrail_name = 'Perplexity')
    plain_results= evaluate_guardrail(perplexity,test_plain, categories, '80_test', methods, guardrail_name = 'Perplexity')
    # Save results and metrics
    dataset_results = plain_results+attacked_results

    save_results(dataset_results,  '80_test', 'Perplexity')
    
if __name__ == '__main__':
    main()
