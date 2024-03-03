import pandas as pd
import numpy as np

train_size = 0.8

file_path = "/data/datasets/mimic3-benchmarks/data/phenotyping/train/listfile.csv"
data_dir = "code/data"


files = pd.read_csv(file_path)

zero_indices = files[files['y_true'] == 0].index.values
one_indices = files[files['y_true'] == 1].index.values

np.random.shuffle(zero_indices)
np.random.shuffle(one_indices)

N = int(len(zero_indices)*train_size)
train_zero_indices = zero_indices[:N]
val_zero_indices = zero_indices[N:]

N = int(len(one_indices)*train_size)
train_one_indices = one_indices[:N]
val_one_indices = one_indices[N:]

train_zero_data = files.iloc[train_zero_indices,:]
val_zero_data = files.iloc[val_zero_indices,:]
train_one_data = files.iloc[train_one_indices,:]
val_one_data = files.iloc[val_one_indices,:]

train_list_file = pd.concat([train_one_data, train_zero_data]).values
val_list_file = pd.concat([val_one_data, val_zero_data]).values

train_indices = list(range(len(train_list_file)))
val_indices = list(range(len(val_list_file)))

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)

train_list_file = train_list_file[train_indices][:]
val_list_file = val_list_file[val_indices][:]

train_list_file = pd.DataFrame(train_list_file,columns=['stay', 'y_true'])
val_list_file = pd.DataFrame(val_list_file,columns=['stay', 'y_true'])

train_list_file.to_csv(f"{data_dir}/train_listfile.csv", index = False)
val_list_file.to_csv(f"{data_dir}/val_list_file.csv", index = False)