import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.utils import *


attack_data = pd.read_csv('/data/Unsafe_Attack.csv')

temp_df, test_df = train_test_split(attack_df, test_size=0.2, random_state=42)