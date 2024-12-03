import httpcore
setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')
import argparse
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from deep_translator import GoogleTranslator
import base64
import random
from datetime import datetime
import pickle
from lingua import Language, LanguageDetectorBuilder
from googletrans import Translator, constants
from attacks.attack_utils import *
from utils.utils import *


detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
translator = Translator()

def decode_base64(s):
    try:
        return base64.b64decode(s).decode("utf-8")
    except Exception:
        return False

def caesar_encrypt(plaintext, shift):
    encrypted_text = ""
    
    for char in plaintext:
        if char.isalpha():  # Only shift alphabetic characters
            shift_start = ord('A') if char.isupper() else ord('a')  # Handle uppercase and lowercase separately
            encrypted_text += chr((ord(char) - shift_start + shift) % 26 + shift_start)
        else:
            encrypted_text += char  # Non-alphabetic characters remain unchanged
    
    return encrypted_text


def load_and_concatenate_test_csv(file_paths, subset, percentage):
    temp = [file+f"_{percentage}_percent/"+f"{subset}.csv" for file in file_paths]
    # Load each CSV into a list of DataFrames
    dfs = [pd.read_csv(file+f"_{percentage}_percent/"+f"{subset}.csv") for file in file_paths]
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# Function to evaluate guardrails
def evaluate_guardrail(model, tokenizer, prompts, categories, dataset_name, attack_name, guardrail_name="DecipherGuard"):
    dataset_results = []
    preds = []
    probs = []
    times = []
    
    base64_time = []
    zulu_time = []
    cipher_time = []
    
    for chat in tqdm(prompts):
        with torch.no_grad():
            start = datetime.now()
            model_input = decode_base64(chat)
            end = datetime.now()
            time_diff = (end - start).total_seconds() * 1000
            base64_time.append(time_diff)
            
            if model_input is not False:
                transformed_chat = transform_dataset_prompt(model_input)
                pred = moderate(transformed_chat, tokenizer, model)
                prob = moderate_probability(transformed_chat, tokenizer, model)
            else:
                
                zulu_start = datetime.now()
                conf = detector.compute_language_confidence_values(chat)[0]
                zulu_end = datetime.now()
                time_diff = (zulu_end - zulu_start).total_seconds() * 1000
                zulu_time.append(time_diff)
                if conf.language.name == "ZULU":
                    model_input = translator.translate(chat, dest="en", src="zu").text
                    transformed_chat = transform_dataset_prompt(model_input)
                    pred = moderate(transformed_chat, tokenizer, model)
                    prob = moderate_probability(transformed_chat, tokenizer, model)
                
                else:
                    cipher_start = datetime.now()
                    potential_probs = []
                    potential_prompts = []
                    for shift in range(0, 26, 1):
                    
                        encrypted = chat
                        if shift != 0:
                            encrypted = caesar_encrypt(chat, shift)
                        
                        potential_prompts.append(encrypted)
                        confidence_values = detector.compute_language_confidence_values(encrypted)
                        for confidence in confidence_values:
                            if confidence.language.name == "ENGLISH":
                                potential_probs.append(confidence.value)
    
                    max_idx = potential_probs.index(max(potential_probs))
                    model_input = potential_prompts[max_idx]
                    cipher_end = datetime.now()
                    time_diff = (cipher_end - cipher_start).total_seconds() * 1000
                    cipher_time.append(time_diff)
                    
                    transformed_chat = transform_dataset_prompt(model_input)
                    pred = moderate(transformed_chat, tokenizer, model)
                    prob = moderate_probability(transformed_chat, tokenizer, model)


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


    return dataset_results, base64_time, zulu_time, cipher_time




# Define a function to transform dataset prompts
def transform_dataset_prompt(prompt):

    return [{"role": "user", "content": prompt}]

# Define the moderation function
def moderate(chat, tokenizer, model, device='cuda:0'):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
def moderate_probability(chat, tokenizer, model, device='cuda:0'):
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

    total_base64_time = []
    total_zulu_time = []
    total_cipher_time = []
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    hf_token = ''  #TODO Replace with your token

    tokenizer = AutoTokenizer.from_pretrained("MickyMike/decipher_lora_5-2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("MickyMike/decipher_lora_5-2", device_map = 'cuda:0',torch_dtype=torch.bfloat16, use_cache=False)
    

    
    attacked_results,base64_time, zulu_time, cipher_time = evaluate_guardrail(model, tokenizer, test_attacked, categories, '80_test', methods, guardrail_name = 'DecipherGuard')

    # Save results and metrics
    dataset_results = attacked_results

    save_results(dataset_results,  '80_test', 'DecipherGuard')
            
    
    
    with open("./finetuned_base64time.pkl", "wb+") as f:
        pickle.dump(base64_time, f)
    with open("./finetuned_zulutime.pkl", "wb+") as f:
        pickle.dump(zulu_time, f)
    with open("./finetuned_ciphertime.pkl", "wb+") as f:
        pickle.dump(cipher_time, f)
if __name__ == '__main__':
    main()
