# Import required packages
import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from config import *

# Simple dataset class
class ORFDataset(Dataset):
    def __init__(self, file_path, debug_samples=None):
        self.df = pd.read_csv(file_path, sep="\t", header=None, names=["id", "seq", "label"])
        if debug_samples is not None:
            self.df = self.df.sample(n=debug_samples, random_state=9).reset_index(drop=True)
            print(f"On DEBUG MODE.")
    
    def __len__(self):
        return len(self.df)   
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return str(row.id), str(row.seq)
        
# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
print("Loading tokenizer and model")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, cache_dir=CACHE_DIR)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, cache_dir=CACHE_DIR).to(device).to(torch.bfloat16)

# Run model in evaluation mode
model.eval()

# Dataset setup
dataset = ORFDataset(SEQ_FILE_PATH, debug_samples=None)
print(f"Number of samples for extraction: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

os.makedirs(os.path.dirname(TRAIN_EMB_FILE), exist_ok=True)

# Embeddings extraction loop
with h5py.File(TRAIN_EMB_FILE, "a") as f:
    for ids, seq in tqdm(dataloader, desc="Extracting embs"):  
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                embeddings = outputs[0].cpu().numpy()       
                
        # Save each seq in the batch individually 
        for i, seq_id in enumerate(ids):
            # Since we padded sequence for batch operation
            # Remove PAD token before saving and [CLS] as well
            # In [1, 1:actual_len-1, :], Keep B, remove [CLS] [PAD] [SEP], keep 768-D
            actual_len = inputs["attention_mask"][i].sum().item()
            actual_emb = embeddings[i, 1:actual_len-1, :]
            
            if seq_id in f:
                del f[seq_id]                                            # Avoid error if rerunning
            
            f.create_dataset(seq_id, data=actual_emb, compression="gzip", chunks=True)
            
        # Clean up
        del inputs, outputs, embeddings
        torch.cuda.empty_cache()