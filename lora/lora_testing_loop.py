import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
import base64
import random
import gc 

torch.autograd.set_detect_anomaly(True)
device = "cuda:0"

# Create a PyTorch Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach().to(device) for key, val in self.encodings.items()}
        return item

def load_and_concatenate_test_csv(file_paths, subset, percentage):
    temp = [file+f"_{percentage}_percent/"+f"{subset}.csv" for file in file_paths]
    # Load each CSV into a list of DataFrames
    dfs = [pd.read_csv(file+f"_{percentage}_percent/"+f"{subset}.csv") for file in file_paths]
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def tokenize_with_template_test(user_inputs, agent_labels):
    input_ids_list = []
    labels_list = []
    
    for user_input, agent_label in tqdm(zip(user_inputs, agent_labels), desc="Tokenizing"):
        # Apply the training chat template
        chat = [
            {"role": "user", "content": user_input},
        ]
        # Format the chat into a single string (without tokenizing yet)
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").squeeze(0).to(device)
                
        # Tokenize the label separately ("unsafe" or "safe")
        label_input = tokenizer(agent_label, padding="max_length", truncation=True, return_tensors="pt", max_length=1)["input_ids"].squeeze(0)
        
        # Prepare the labels tensor with -100 (ignore index for loss calculation)
        labels = torch.full_like(input_ids, -100)
        
        # Find the index of the first padding token in the input (where the conversation ends)
        # Assuming padding is done with the token ID 0, you can adjust if using a different pad_token_id
        first_pad_index = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]  # First pad index
        
        # Insert the "unsafe"/"safe" label at the first pad position (or the end of the prompt)
        labels[first_pad_index] = label_input  # Place the label token
                
        # Append the input_ids and labels to their respective lists
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        
    encodings = {
        "input_ids": input_ids_list,
        "labels": labels_list
    }
    return encodings



    
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
test_texts = test_df["transformed_prompt"].tolist()
# All outputs are labeled as 'unsafe' (in chat context, we still follow the chat template)
test_labels = ["unsafe"] * len(test_texts)


with open('./data/data/alpaca_small.json') as f:
    alpaca = json.load(f)
    alpaca_texts = [d['instruction'] for d in alpaca][:2000]
alpaca_labels = ["safe"] * len(alpaca_texts)
alpaca_df = pd.DataFrame({'prompts':alpaca_texts})



for p in [1,3,5,7,10,20]:
    for ite in [1,2,3,4,5]:
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
        # Initialize tokenizer and model

        tokenizer = AutoTokenizer.from_pretrained(f"MickyMike/lora_{p}-{ite}")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # model = AutoModelForCausalLM.from_pretrained("MickyMike/DecipherGuard2", device_map="cuda:0", torch_dtype=torch.bfloat16, use_cache=False)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B", # meta-llama/Llama-Guard-3-8B
                                                    device_map=device,
                                                    torch_dtype=torch.bfloat16,
                                                    use_cache=False)
        # load PEFT model
        model = PeftModel.from_pretrained(model,
                                        f"MickyMike/lora_{p}-{ite}",
                                        device_map=device,
                                        torch_dtype=torch.bfloat16,
                                        use_cache=False)
        # Merge LoRA weights into the base model 
        model = model.merge_and_unload()
        # Test the model
        model.eval()

        # encode test data 
        test_encodings = tokenize_with_template_test(test_texts, test_labels)
        test_dataset = TextDataset(test_encodings)
        # Create DataLoader for batching
        test_dataloader = DataLoader(test_dataset, batch_size=1)

        # encode safe prompts
        alpaca_encodings = tokenize_with_template_test(alpaca_texts, alpaca_labels)
        alpaca_dataset = TextDataset(alpaca_encodings)
        # Create DataLoader for batching
        alpaca_dataloader = DataLoader(alpaca_dataset, batch_size=1)
        # Loop through the test dataloader
        count = 0
        correct_count = 0
        predictions = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):
                input_ids = batch["input_ids"]
                # Generate outputs using the model
                try:
                    generated_outputs = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
                except:
                    predictions.append(-1)
                    continue
                prompt_len = input_ids.shape[-1]
                # Decode and print each generated output
                for input_id, generated_output in zip(input_ids, generated_outputs):
                    # Decode the input for reference
                    input_text = tokenizer.decode(input_id, skip_special_tokens=True)
                    
                    # Decode the generated output
                    output_text = tokenizer.decode(generated_output[prompt_len:], skip_special_tokens=True)
                    count += 1
                    if "unsafe" in output_text:
                        correct_count += 1
                        predictions.append(1)
                    else:
                        predictions.append(0)
                    print("DSR: ", round(correct_count/count, 2))
                    print(f"{correct_count}/{count}")
        print(f"DSR for model with {p} percent training iteration {ite}: ", round(correct_count/count, 2))
        print(f"{correct_count}/{count}")
        test_df[f"predictions_{p}-{ite}"] = predictions
        test_df.to_csv(f"./finetune_results_{p}.csv")


        # safe prompts
        count = 0
        correct_count = 0
        predictions = []

        with torch.no_grad():
            for batch in tqdm(alpaca_dataloader, desc="Testing"):
                input_ids = batch["input_ids"]
                # Generate outputs using the model
                try:
                    generated_outputs = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
                except:
                    predictions.append(-1)
                    continue
                prompt_len = input_ids.shape[-1]
                # Decode and print each generated output
                for input_id, generated_output in zip(input_ids, generated_outputs):
                    # Decode the input for reference
                    input_text = tokenizer.decode(input_id, skip_special_tokens=True)
                    # Decode the generated output
                    output_text = tokenizer.decode(generated_output[prompt_len:], skip_special_tokens=True)
                    count += 1
                    if "unsafe" in output_text:
                        correct_count += 1
                        predictions.append(1)
                    else:
                        predictions.append(0)
                    print("FAR: ", round(correct_count/count, 2))
                    print(f"{correct_count}/{count}")
                
        print("FAR: ", round(correct_count/count, 2))
        print(f"{correct_count}/{count}")
        alpaca_df[f"predictions_{p}-{ite}"] = predictions
        alpaca_df.to_csv(f"./safe_prompt_results_{p}.csv")


print("done")