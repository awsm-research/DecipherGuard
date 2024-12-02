import argparse
import os
import json
import pandas as pd
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from deep_translator import GoogleTranslator
import base64
import random
from attacks.attack_utils import *
from openai import OpenAI
from utils.utils import *
client = OpenAI(api_key='') #TODO Replace with your token



def load_and_concatenate_test_csv(file_paths, subset, percentage):
    temp = [file+f"_{percentage}_percent/"+f"{subset}.csv" for file in file_paths]
    # Load each CSV into a list of DataFrames
    dfs = [pd.read_csv(file+f"_{percentage}_percent/"+f"{subset}.csv") for file in file_paths]
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df




# Function to evaluate guardrails
def evaluate_guardrail(prompts, categories, dataset_name, attack_name, guardrail_name="OpenAI"):
    dataset_results = []
    preds = []
    probs = []
    responses = []
    batch_size = max(1, len(prompts) // 10)  # Calculate 10% of the data or at least 1
    num_batches = (len(prompts) + batch_size - 1) // batch_size  # Ensure all data is processed

    for batch in range(num_batches):
        # Process prompts in the current batch
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        for prompt in tqdm(batch_prompts):

            response = client.moderations.create(
                model="omni-moderation-latest",
                input=[prompt]
            )
            responses.append(response)
            time.sleep(1)
        
        # time.sleep(60)  # Sleep for 60 seconds after processing the batc

            
    for chat in range(len(prompts)):
        if responses[chat].results[0].flagged == False:
            preds.append(0)
        else:
            preds.append(1)
        

    for resp in responses:
        try:
            scores = list(resp.results[0].category_scores)
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            probs.append(sorted_scores[0][1])
        except (TypeError, AttributeError, IndexError) as e:
            print(f"Error processing response: {e}")
            probs.append(None) 
    

    for idx, prompt in enumerate(prompts):
        result = {
            'prompt': prompt,
            'category': categories[idx],
            'ID': f'{guardrail_name}_{dataset_name}_{attack_name[idx]}_{idx}',
            'attack_method': attack_name[idx],
            'guardrail': guardrail_name,
            'probability': probs[idx],  
            'pred': preds[idx],
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


    all_results = []

    # transformed_prompts = transform_dataset_prompt(attacked_prompts)
    attacked_results = evaluate_guardrail(test_attacked, categories, '80_test', methods, guardrail_name = 'OpenAI')
    plain_results = evaluate_guardrail(test_plain,  categories, '80_test', methods, guardrail_name = 'OpenAI')
    # Save results and metrics
    dataset_results = plain_results+attacked_results
    save_results(dataset_results,  '80_test', 'OpenAI')
            
 

if __name__ == '__main__':
    main()
