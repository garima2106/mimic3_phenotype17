MAX_LEN = 256
batch_size = 32
d_model = 128
num_heads = 8
N = 6
num_variables = 17 
num_variables += 1 #for no variable embedding while doing padding
d_ff = 512
epochs = 75
learning_rate = 1e-5
drop_out = 0.1
sinusoidal = True
th_val_roc = 0.84
th_val_pr = 0.48
num_classes = 25
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import pandas as pd
import numpy as np
from utils import MimicDataSetPhenotyping, calculate_auc_roc, calculate_auc_prc
#pd.set_option('future.no_silent_downcasting',True)
pd.set_option('mode.use_inf_as_na', True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F

from model import Model
from tqdm import tqdm
from normalizer import Normalizer
from categorizer import Categorizer

import matplotlib.pyplot as plt

import seaborn as sns

    
    
train_data_path = "/data/datasets/mimic3-benchmarks/data/phenotyping/train_listfile.csv"
val_data_path = "/data/datasets/mimic3-benchmarks/data/phenotyping/val_listfile.csv"

data_dir = "/data/datasets/mimic3-benchmarks/data/phenotyping/train/"


import pickle

with open('normalizer.pkl', 'rb') as file:
    normalizer = pickle.load(file)

with open('categorizer.pkl', 'rb') as file:
    categorizer = pickle.load(file)
    

mean_variance = normalizer.mean_var_dict
cat_dict = categorizer.category_dict


train_ds = MimicDataSetPhenotyping(data_dir, train_data_path, mean_variance, cat_dict, 'training', MAX_LEN)
val_ds = MimicDataSetPhenotyping(data_dir, val_data_path, mean_variance, cat_dict, 'validation', MAX_LEN)



train_dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True)

model = Model(d_model, num_heads, d_ff, num_classes, N, sinusoidal).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()

accumulation_steps = 2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
best_roc_auc_macro = 0.0
best_prc_auc_macro = 0.0
early_stop_counter = 0
patience = 10

for epoch in range(epochs):
    running_loss = 0.0
    for i, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)):
        inp = batch['encoder_input']
        mask = batch['encoder_mask']
        y = batch['label']
        outputs = model(inp, mask)
        loss = criterion(outputs.view(-1), y.float().view(-1))
        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        
    # Learning rate scheduling
    scheduler.step()
    print("Epoch {}, Loss: {:.3f}".format(epoch + 1, running_loss / len(train_dataloader)))
    #print("Loss", loss.item())
    roc_auc_micro, roc_auc_macro = calculate_auc_roc(model, val_dataloader)
    prc_auc_micro, prc_auc_macro = calculate_auc_prc(model, val_dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Micro AUC-ROC: {roc_auc_micro:.3f}, Macro AUC-ROC: {roc_auc_macro:.3f}')
    print(f'Epoch {epoch + 1}/{epochs}, Micro AUC-PRC: {prc_auc_micro:.3f}, Macro AUC-PRC: {prc_auc_macro:.3f}')
    # Check for improvement in Macro AUC-ROC and Macro AUC-PRC
    if roc_auc_macro > best_roc_auc_macro or prc_auc_macro > best_prc_auc_macro:
        best_roc_auc_macro = max(roc_auc_macro, best_roc_auc_macro)
        best_prc_auc_macro = max(prc_auc_macro, best_prc_auc_macro)
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    # Check if early stopping criteria met
    if early_stop_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1} as no improvement observed in {patience} epochs.')
        break

# Save the model
file_path = f"model_maxlen{MAX_LEN}_batch{batch_size}_dmodel{d_model}_heads{num_heads}_N{N}_vars{num_variables}_dff{d_ff}_epochs{epoch + 1}_lr{learning_rate}_dropout{drop_out}_sinusoidal{sinusoidal}_testing.pth"
torch.save(model.state_dict(), "models/" + file_path)
