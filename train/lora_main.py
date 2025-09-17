import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict
from tqdm import tqdm
import os
from transformers import get_scheduler
import argparse
import logging
import numpy as np
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123456,
                    help="random seed for initialization")

parser.add_argument("--data_dir", default="data", type=str,
                    help="Folder name for the data.")
parser.add_argument("--do_train", action='store_true', default=False,
                    help="Whether to train the model.")
parser.add_argument("--do_test", action='store_true', default=False,
                    help="Whether to run inference.")
parser.add_argument("--do_ablation", action='store_true', default=False,
                    help="Whether to do ablation study on the amount of training data.")
parser.add_argument("--training_proportion", default=None, type=int,
                    help="")

parser.add_argument("--model_name_or_path", default="meta-llama/Llama-Guard-3-8B", type=str,
                    help="The model checkpoint for weights initialization.")
parser.add_argument("--saved_model_name", default=None, type=str,
                    help="Name for the best saved model.")
parser.add_argument("--test_result_file", default=None, type=str,
                    help="File name and path for writing testing results.")

parser.add_argument("--learning_rate", default=1e-3, type=float,
                    help="LoRA parameters")
parser.add_argument("--epochs", default=1, type=int,
                    help="LoRA parameters")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Maximum number of subword tokens per input.")

parser.add_argument("--lora_r", default=8, type=int,
                    help="LoRA parameters")
parser.add_argument("--lora_alpha", default=32, type=int,
                    help="LoRA parameters")
parser.add_argument("--lora_dropout", default=0.1, type=float,
                    help="LoRA parameters")

parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for train/val/test")
parser.add_argument("--max_train_input_length", default=2048, type=int,
                    help="Maximum number of subword tokens per input.")
parser.add_argument("--max_new_tokens", default=100, type=int,
                    help="Generation parameters")

args = parser.parse_args()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

set_seed(args)

# Create a PyTorch Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

def load_and_concatenate_csv(file_paths, subset):
    # Load each CSV into a list of DataFrames
    dfs = [pd.read_csv(file+f"{subset}.csv") for file in file_paths]
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def tokenize_with_template(user_inputs, agent_labels):
    input_ids_list = []
    labels_list = []

    for user_input, agent_label in tqdm(zip(user_inputs, agent_labels), desc="Tokenizing"):
        # Apply the training chat template
        chat = [
            {"role": "user", "content": user_input},
        ]
        
        # Format the chat into a single string (without tokenizing yet)
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", max_length=args.max_train_input_length
                                                  , padding="max_length", truncation=True).squeeze(0)
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

    # Stack all tokenized inputs and labels into single tensors
    encodings = {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list)
    }
    
    return encodings

def tokenize_with_template_test(user_inputs, agent_labels):
    input_ids_list = []
    labels_list = []

    for user_input, agent_label in tqdm(zip(user_inputs, agent_labels), desc="Tokenizing"):
        # Apply the training chat template
        chat = [
            {"role": "user", "content": user_input},
        ]
        
        # Format the chat into a single string (without tokenizing yet)
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").squeeze(0)
                
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
                    f"./{args.data_dir}/AIM_data_{args.training_proportion}_percent/",
                    f"./{args.data_dir}/base64_data_{args.training_proportion}_percent/",
                    f"./{args.data_dir}/caesar_cipher_data_{args.training_proportion}_percent/",
                    f"./{args.data_dir}/CC_data_{args.training_proportion}_percent/",
                    f"./{args.data_dir}/combination_data_{args.training_proportion}_percent/",
                    f"./{args.data_dir}/DAN_data_{args.training_proportion}_percent/",
                    f"./{args.data_dir}/deepInception_data_{args.training_proportion}_percent/",
                    f"./{args.data_dir}/dual_use_data_{args.training_proportion}_percent/",
                    f"./{args.data_dir}/self_cipher_data_{args.training_proportion}_percent/",
                    f"./{args.data_dir}/zulu_data_{args.training_proportion}_percent/",
                  ] 
if args.do_train:
    train_df = load_and_concatenate_csv(all_file_paths, subset="train")
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_texts = train_df["transformed_prompt"].tolist()
    # all outputs are labeled as 'unsafe'
    train_labels = ["unsafe"] * len(train_texts)

if args.do_test:
    test_df = load_and_concatenate_csv(all_file_paths, subset="test")    
    test_texts = test_df["transformed_prompt"].tolist()
    # all outputs are labeled as 'unsafe'
    test_labels = ["unsafe"] * len(test_texts)

# load tokenizer and model
if args.do_train:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) # meta-llama/Llama-Guard-3-8B
elif args.do_test:
    tokenizer = AutoTokenizer.from_pretrained(args.saved_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.do_train:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                use_cache=False)
    # Apply LoRA configuration
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
    ### Default: trainable params: 4,718,592 || all params: 8,034,979,840 || trainable%: 0.0587 ###
    lora_config = LoraConfig(task_type="CAUSAL_LM",
                            inference_mode=False,
                            r=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout,
                            target_modules=target_modules,
                            bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    exit()

# Create DataLoader for batching
if args.do_train:
    train_encodings = tokenize_with_template(train_texts, train_labels)
    train_dataset = TextDataset(train_encodings)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

if args.do_test:
    test_encodings = tokenize_with_template_test(test_texts, test_labels)
    test_dataset = TextDataset(test_encodings)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

# start training
if args.do_train:
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    num_train_steps = len(train_dataloader) * args.epochs  # Number of batches * epochs
    lr_scheduler = get_scheduler(
                                    name="constant",
                                    optimizer=optimizer,
                                    num_warmup_steps=0,  # No warmup for constant scheduler
                                    num_training_steps=num_train_steps  # Total training steps
                                )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader) * args.batch_size)
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Train batch size = %d", args.batch_size)
    logger.info("  Test batch size = %d", args.batch_size)
    logger.info("  Gradient Clipping = %d", args.max_grad_norm)
    logger.info("  Total optimization steps = %d", num_train_steps)
    
    # Training loop
    best_loss = 1e6
    model.train()
    for epoch in range(args.epochs):  # Number of epochs
        total_loss = 0
        tr_loss = 0
        tr_num = 0
        avg_loss = 0
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        for batch in bar:
            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()  # Update the learning rate

            tr_loss = loss.item()
            total_loss += tr_loss
            tr_num += 1
            avg_loss = round(total_loss/tr_num, 5)
            
            bar.set_description("epoch {} loss {}".format(epoch, avg_loss))

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Avg Train Loss: {avg_train_loss}")
        
        # Save the model
        best_model_path = f"./{args.saved_model_name}"
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        logger.info("Saving model checkpoint to %s", best_model_path)
        
        model.train()

if args.do_test:
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataloader)*args.batch_size)
    logger.info("  Batch size = %d", args.batch_size)
    
    # 1 GPU for testing
    device = "cuda:0"
    
    # load base model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B",
                                                device_map=device,
                                                torch_dtype=torch.bfloat16,
                                                use_cache=False)
    model = PeftModel.from_pretrained(model,
                                    args.saved_model_name,
                                    device_map=device,
                                    torch_dtype=torch.bfloat16,
                                    use_cache=False)
    
    model = model.merge_and_unload()
    # Merge LoRA weights into the base model
    model.eval()
    
    # Loop through the test dataloader
    count = 0
    correct_count = 0
    predictions = []
    with torch.no_grad():
        test_bar = tqdm(test_dataloader, total=len(test_dataloader))
        for batch in test_bar:
            input_ids = batch["input_ids"].to(device) # use one GPU for testing (cuda:0)
            # Generate outputs using the model
            try:
                generated_outputs = model.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens, pad_token_id=0)
            except:
                # invalid input
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
                dsr = round(correct_count/count, 2)
                test_bar.set_description(f"Test DSR: {dsr} | {correct_count}/{count}")
    
    logger.info(f"***** Test Results *****")
    logger.info(f"DSR: {round(correct_count/count, 2)}")
    logger.info(f"{correct_count}/{count}")
    
    logger.info(f"***** Writing Testing Results to {args.test_result_file} *****")
    test_df["predictions"] = predictions
    test_df.to_csv(args.test_result_file)
    logger.info("done.")