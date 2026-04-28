import json
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
import os
# Function to load selected datasets
def load_selected_datasets(selected_datasets):
    datasets = {}
    categories = {}

    if 'DNA' in selected_datasets:
        DNA = load_dataset('LibrAI/do-not-answer')
        DNA_prompts = [item['question'] for item in DNA['train'] if item['risk_area'] in ['Malicious Uses', 'Discrimination, Exclusion, Toxicity, Hateful, Offensive']]
        categories['DNA'] = [item['types_of_harm'] for item in DNA['train'] if item['risk_area'] in ['Malicious Uses', 'Discrimination, Exclusion, Toxicity, Hateful, Offensive']]
        datasets['DNA'] = DNA_prompts
    if 'advBench' in selected_datasets:
        advBench = load_dataset('walledai/AdvBench')
        datasets['advBench'] = advBench['train']['prompt']
        categories['advBench'] = ['' for item in datasets['advBench']]

    if 'forbidden_questions' in selected_datasets:
        forbidden_questions = pd.read_csv('./data/forbidden_question_set.csv')
        datasets['forbidden_questions'] = forbidden_questions['question'].tolist()
        categories['forbidden_questions'] = forbidden_questions['content_policy_name'].tolist()


    if 'HarmfulQA' in selected_datasets:
        HarmfulQA = load_dataset("declare-lab/CategoricalHarmfulQA")
        datasets['HarmfulQA'] = HarmfulQA['en']['Question']
        categories['HarmfulQA'] = HarmfulQA['en']['Category']
    if 'BeaverTails' in selected_datasets:
        beaverTails = load_dataset('PKU-Alignment/BeaverTails')['30k_test'].to_pandas()
        datasets['BeaverTails'] = beaverTails[beaverTails['is_safe'] == True]['prompt'].tolist()
        categories['BeaverTails'] = ['-' for i in datasets['BeaverTails']]
    if 'alpaca' in selected_datasets:
        with open('./data/alpaca_small.json') as f:
            alpaca = json.load(f)
            datasets['alpaca'] = [d['instruction'] for d in alpaca][:2000]
            categories['alpaca'] = ['-' for i in datasets['alpaca']]


    return datasets, categories


# Function to compute evaluation metrics
def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    pr, re, _ = precision_recall_curve(labels, preds)
    auprc = auc(re, pr)
    
    attack_success_count = sum((label == 1 and pred == 0) for label, pred in zip(labels, preds))
    attack_success_rate = attack_success_count / len(labels)

    return accuracy, precision, recall, f1, auprc, attack_success_rate

# Function to save results and metrics
def save_results_and_metrics(results, metrics, attack_name, guardrail_name):
    output_dir = 'results/'+guardrail_name
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    results_df = pd.DataFrame(results, columns=['prompt', 'transformed_prompt', 'category', 'ID', 'attack_method', 'guardrail', 'probability', 'label'])
    output_file = os.path.join(output_dir, f'{guardrail_name}_{attack_name}_results.csv')
    results_df.to_csv(output_file, index=False)

    # Save metrics
    # metrics_df = pd.DataFrame(metrics, columns=['Attack', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUPRC', 'Attack Success Rate'])
    # output_file = os.path.join(output_dir, f'{guardrail_name}_{attack_name}_{dataset_name}_metrics.csv')
    # metrics_df.to_csv(output_file, index=False)

    print(f"Results and metrics saved to {output_file}")

def save_results(results,  attack_name, guardrail_name):
    output_dir = 'results/'+guardrail_name
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    results_df = pd.DataFrame(results, columns=['prompt', 'transformed_prompt', 'category', 'ID', 'attack_method', 'guardrail', 'probability','pred'])
    output_file = os.path.join(output_dir, f'{guardrail_name}_{attack_name}_results.csv')
    results_df.to_csv(output_file, index=False)


    print(f"Results and metrics saved to {output_file}")


def save_results_and_metrics_time(results, metrics, attack_name, guardrail_name):
    output_dir = 'results/'+guardrail_name
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    results_df = pd.DataFrame(results, columns=['prompt', 'transformed_prompt', 'category', 'ID', 'attack_method', 'guardrail', 'probability', 'label', 'time'])
    output_file = os.path.join(output_dir, f'{guardrail_name}_{attack_name}_results.csv')
    results_df.to_csv(output_file, index=False)

    print(f"Results and metrics saved to {output_file}")