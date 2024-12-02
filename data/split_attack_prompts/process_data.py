import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Read the CSV file
df = pd.read_csv("./LlamaGuard_Unsafe_Attack.csv")

# List of all attack methods
all_attacks = ['caesar_cipher', 'zulu', 'dual_use', 'base64', 'AIM', 'deepInception', 'self_cipher', 'DAN', 'CC', 'combination']

training_proportion = [1,3,5,7,10]

for t in training_proportion:
    # Process each attack method
    for attack in all_attacks:
        # Filter the dataframe for the current attack method
        attack_df = df[df["attack_method"] == attack].reset_index(drop=True)
        
        temp_df, test_df = train_test_split(attack_df, test_size=0.8, random_state=42)  # 80% for testing
        
        x = t/20
        
        train_df, _ = train_test_split(temp_df, test_size=1-x, random_state=42)
        
        print(x*0.2)
        
        # Create directory for the current attack method
        dir_name = f"./{attack}_data_{t}_percent"
        os.makedirs(dir_name, exist_ok=True)
        
        # Write the dataframes to separate CSV files
        train_df.to_csv(f"{dir_name}/train.csv", index=False)
        test_df.to_csv(f"{dir_name}/test.csv", index=False)
        
        print(f"Data for {attack} saved in {dir_name}")
