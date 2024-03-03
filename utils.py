import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MimicDataSetPhenotyping(Dataset):
    def __init__(self, data_dir, csv_file, mean_variance, cat_dict, mode, seq_len, pad_value=0, device='cuda'):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.seq_len = seq_len
        self.mode = mode
        self.data_df = pd.read_csv(csv_file)
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.device = device
        self.cat_dict = cat_dict

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        path = self.data_dir + self.data_df['stay'][idx]
        data = pd.read_csv(path)

        # Replace values
        data.replace(['ERROR', 'no data', '.', '-', '/', 'VERIFIED', 'CLOTTED', "*", 'ERROR DISREGARD PREVIOUS RESULT OF 32', 'DISREGARD PREVIOUSLY REPORTED 33'], np.nan, inplace=True)

        # Extract values
        sample = self.extract(data)

        if len(sample[0]) >= self.seq_len:
            sample = [x[-self.seq_len:] for x in sample]

        num_pad_tokens = self.seq_len - len(sample[0])

        variable_input = torch.cat([
            torch.tensor(sample[2], dtype=torch.int64),
            torch.full((num_pad_tokens,), self.pad_value, dtype=torch.int64)
        ]).to(self.device)

        value_input = torch.cat([
            torch.tensor(sample[1], dtype=torch.float),
            torch.full((num_pad_tokens,), self.pad_value, dtype=torch.float)
        ]).to(self.device)

        time_input = torch.cat([
            torch.tensor(sample[0], dtype=torch.float) - torch.tensor(sample[0], dtype=torch.float).min(),
            torch.full((num_pad_tokens,), self.pad_value, dtype=torch.float)
        ]).to(self.device)

        variables = sample[3] + ['pad token'] * num_pad_tokens

        cols = self.data_df.columns[2:]
        label = torch.tensor(self.data_df[cols].values[idx], dtype=torch.int64).to(self.device)

        return {
            "encoder_input": [time_input, variable_input, value_input],
            "encoder_mask": (variable_input != self.pad_value).unsqueeze(0).int(),
            "variables": variables,
            "label": label
        }

    def extract(self, data):
        sample = [[], [], [], []]
        id_name_dict = {i: data.columns[i] for i in range(len(data.columns))}

        for i in range(data.shape[0]):
            time = data.iloc[i, 0]
            for j in range(1, data.shape[1]):
                if not pd.isna(data.iloc[i, j]):
                    if id_name_dict[j] in self.cat_dict:
                        sample[0].append(time)
                        sample[1].append(self.cat_dict[id_name_dict[j]][data.iloc[i, j]])
                        sample[2].append(j)
                        sample[3].append(id_name_dict[j])
                    else:
                        mean = self.mean_variance[id_name_dict[j]]['mean']
                        var = self.mean_variance[id_name_dict[j]]['variance']
                        val = (float(data.iloc[i, j]) - mean) / var
                        sample[0].append(time)
                        sample[1].append(val)
                        sample[2].append(j)
                        sample[3].append(id_name_dict[j])
        return sample
    def isNAN(self, val):
        return val!=val


import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

def calculate_auc_roc(model, data_loader):
    model.eval()
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for inputs in tqdm(data_loader, leave=False):
            logits = model(inputs['encoder_input'], inputs['encoder_mask'])
            probabilities = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probabilities
            labels = inputs['label']
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probabilities_all = np.concatenate(all_probabilities, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)

    # Micro AUC-ROC
    roc_auc_micro = roc_auc_score(labels_all, probabilities_all, average='micro')

    # Macro AUC-ROC
    roc_auc_macro = roc_auc_score(labels_all, probabilities_all, average='macro')

    return roc_auc_micro, roc_auc_macro

def calculate_auc_prc(model, data_loader):
    model.eval()
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for inputs in tqdm(data_loader, leave=False):
            logits = model(inputs['encoder_input'], inputs['encoder_mask'])
            probabilities = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probabilities
            labels = inputs['label']
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probabilities_all = np.concatenate(all_probabilities, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)

    # Micro AUC-PRC
    prc_auc_micro = average_precision_score(labels_all, probabilities_all, average='micro')

    # Macro AUC-PRC
    prc_auc_macro = average_precision_score(labels_all, probabilities_all, average='macro')

    return prc_auc_micro, prc_auc_macro
# from sklearn.metrics import roc_auc_score, average_precision_score
# from tqdm import tqdm

# def calculate_auc_roc(model, data_loader):
#     model.eval()
#     all_probabilities = []
#     all_labels = []

#     with torch.no_grad():
#         for inputs in tqdm(data_loader, leave=False):
#             outputs = model(inputs['encoder_input'], inputs['encoder_mask'])
#             labels = inputs['label']
#             all_probabilities.append(outputs.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())

#     logits_all = np.concatenate(all_probabilities, axis=0)
#     labels_all = np.concatenate(all_labels, axis=0)

#     # Micro AUC-ROC
#     roc_auc_micro = roc_auc_score(labels_all, logits_all, average='micro')

#     # Macro AUC-ROC
#     roc_auc_macro = roc_auc_score(labels_all, logits_all, average='macro')

#     return roc_auc_micro, roc_auc_macro


# def calculate_auc_prc(model, data_loader):
#     model.eval()
#     all_probabilities = []
#     all_labels = []

#     with torch.no_grad():
#         for inputs in tqdm(data_loader, leave=False):
#             outputs = model(inputs['encoder_input'], inputs['encoder_mask'])
#             labels = inputs['label']
#             all_probabilities.append(outputs.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())

#     logits_all = np.concatenate(all_probabilities, axis=0)
#     labels_all = np.concatenate(all_labels, axis=0)

#     # Micro AUC-PRC
#     prc_auc_micro = average_precision_score(labels_all, logits_all, average='micro')

#     # Macro AUC-PRC
#     prc_auc_macro = average_precision_score(labels_all, logits_all, average='macro')

#     return prc_auc_micro, prc_auc_macro

'''from sklearn.metrics import roc_auc_score
from tqdm import tqdm
def calculate_roc_auc(model, data_loader, num_classes=25):
    model.eval()
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for inputs in tqdm(data_loader, leave=False):
            outputs = model(inputs['encoder_input'], inputs['encoder_mask'])
            labels = inputs['label']
            logits = torch.sigmoid(outputs)
            
            all_probabilities.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probabilities_all = np.concatenate(all_probabilities)
    labels_all = np.concatenate(all_labels)
    
    probabilities_all = np.nan_to_num(probabilities_all, nan=0.0)
    labels_all = np.nan_to_num(labels_all, nan=0.0)
    
    probabilities_all = np.clip(probabilities_all, a_min=-1e7, a_max=1e7)
    roc_auc_scores = [roc_auc_score(labels_all[:, i], probabilities_all[:, i]) for i in range(num_classes)]
    roc_auc = np.mean(roc_auc_scores)
    return roc_auc

from sklearn.metrics import average_precision_score
from tqdm import tqdm

def calculate_auc_prc(model, data_loader, num_classes=25):
    model.eval()
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for inputs in tqdm(data_loader, leave=False):
            outputs = model(inputs['encoder_input'], inputs['encoder_mask'])
            labels = inputs['label']
            logits = torch.sigmoid(outputs)

            all_probabilities.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probabilities_all = np.concatenate(all_probabilities)
    labels_all = np.concatenate(all_labels)
    probabilities_all = np.nan_to_num(probabilities_all, nan=0.0)
    labels_all = np.nan_to_num(labels_all, nan=0.0)
    
    probabilities_all = np.clip(probabilities_all, a_min=-1e7, a_max=1e7)

    auc_prc_scores = [average_precision_score(labels_all[:, i], probabilities_all[:, i]) for i in range(num_classes)]
    auc_prc = np.mean(auc_prc_scores)
    return auc_prc'''


# def get_mean_var(data, data_dir):
#     categorical_variables = ['Glascow coma scale eye opening', 
#                                  'Glascow coma scale motor response', 
#                                  'Glascow coma scale verbal response']
#     sample_path = data_dir + data['stay'][0]
#     id_name_dict = {}
#     df = pd.read_csv(sample_path)
#     df.drop(labels=categorical_variables, axis=1, inplace=True)
#     for i in range(len(df.columns)):
#         id_name_dict[i] = df.columns[i]
#     variable_values = {k : [] for k in df.columns[1:]}
#     for sample_path in tqdm(data['stay']):
#         sample_path = data_dir+sample_path
#         df = pd.read_csv(sample_path)
#         values = df.values
#         df.drop(labels=categorical_variables, axis=1, inplace=True)
#         cols = df.columns[1:]
#         df = df[cols]
#         values = df.values
#         for i in range(values.shape[0]):
#             for j in range(values.shape[1]):
#                 try :
#                     np.isnan(values[i][j])
#                 except:
#                     print(values[i][j])
#                 if np.isnan(values[i][j]) == False:
#                     variable_values[id_name_dict[j+1]].append(values[i][j])
#     result_dict = {}
#     for feature, values in variable_values.items():
#         mean_value = np.mean(values)
#         variance_value = np.var(values)
#         result_dict[feature] = {'mean': mean_value, 'variance': variance_value}
#     return result_dict


