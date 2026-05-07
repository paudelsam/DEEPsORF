# Import required  modules
import h5py
import pandas as pd

import torch
from torch.utils.data import Dataset

class ORFDataset(Dataset):
    def __init__(self, id_label_file, emb_file, debug_samples=None):
        super().__init__()
        self.id_label = pd.read_csv(id_label_file, sep="\t", header=None, names=["id", "label"])
        
        # FOR DEBUGGING
        if debug_samples is not None:
            self.id_label = self.id_label.sample(n=debug_samples, random_state=42).reset_index(drop=True)
            print(f"In DEBUG mode. Debug samples: {debug_samples}")
        # File path only. We aren't opening file here
        self.emb_file = emb_file
        self.emb_store = None
        
        
    def __len__(self):
        return len(self.id_label)
        
    def __getitem__(self, index):
        if self.emb_store is None:
            self.emb_store = h5py.File(self.emb_file, "r")
            
        row = self.id_label.iloc[index]
        seq_id = row.id
        label = torch.tensor(row.label, dtype=torch.float32)
        
        # Direct indexing from already opened file
        emb = torch.tensor(self.emb_store[seq_id][:], dtype=torch.float32)
        
        return seq_id, emb, label
    
    # Make dataset safe for DataLoader with num_workers>0
    def __getstate__(self):
        state = self.__dict__.copy()
        state["emb_store"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

