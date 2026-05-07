# Import required  modules
import random
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

# Seed everything 
def seed_everything(seed=9):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Collate function for batching
def collate_function(batch):
    id, emb, labels = zip(*batch)
    
    # Pad sequences (emb) to maximum length in the batch
    lengths = torch.tensor([e.shape[0] for e in emb], dtype=torch.long)
    emb_padded = pad_sequence(emb, batch_first=True)                      # [B, L, E]
    
    # Masks using embs length
    max_len = emb_padded.size(1)
    arr = torch.arange(max_len, device=emb_padded.device)
    mask = (arr[None, :] < lengths[:, None]).unsqueeze(1)                  # [B, 1, L]
    
    labels = torch.stack(labels)
    
    return {"embeddings": emb_padded,
            "mask": mask,
            "labels": labels}
        
# Label smoothing
# Alpha is written as epsilon here
def get_smoothed_labels(labels, epsilon, classes):
    smoothed_labels = labels.float() * (1 - epsilon) + epsilon / classes
    return smoothed_labels

# Layer-wise learning rate decay for DNABERT2 fine-tuning
def get_layerwise_optimizer_params(model, base_lr, weight_decay):
    """Create parameter groups with different learning rates for different components"""
    
    no_decay = ["bias", "LayerNorm", "GroupNorm", "layer_norm", "group_norm"]
    
    param_groups = []

    
    # CNN branch parameters - medium LR
    cnn_params_decay = [p for n, p in model.cnn_branch.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad]
    cnn_params_no_decay = [p for n, p in model.cnn_branch.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad]
    
    if cnn_params_decay:
        param_groups.append({
            "params": cnn_params_decay,
            "lr": base_lr * 0.5,
            "weight_decay": weight_decay,
            "name": "cnn_decay"
        })
    
    if cnn_params_no_decay:
        param_groups.append({
            "params": cnn_params_no_decay,
            "lr": base_lr * 0.5,
            "weight_decay": 0.0,
            "name": "cnn_no_decay"
        })
   
    # BiGRU branch parameters - medium LR
    gru_params_decay = [p for n, p in model.bigru_branch.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad]
    gru_params_no_decay = [p for n, p in model.bigru_branch.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad]
    
    if gru_params_decay:
        param_groups.append({
            "params": gru_params_decay,
            "lr": base_lr * 0.5,
            "weight_decay": weight_decay,
            "name": "gru_decay"
        })
    
    if gru_params_no_decay:
        param_groups.append({
            "params": gru_params_no_decay,
            "lr": base_lr * 0.5,
            "weight_decay": 0.0,
            "name": "gru_no_decay"
        })
    
    # MLP classifier parameters - lowest LR
    mlp_params_decay = [p for n, p in model.mlp.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad]
    mlp_params_no_decay = [p for n, p in model.mlp.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad]
    
    if mlp_params_decay:
        param_groups.append({
            "params": mlp_params_decay,
            "lr": base_lr * 0.1,
            "weight_decay": weight_decay,
            "name": "mlp_decay"
        })
    
    if mlp_params_no_decay:
        param_groups.append({
            "params": mlp_params_no_decay,
            "lr": base_lr * 0.1,
            "weight_decay": 0.0,
            "name": "mlp_no_decay"
        })
    
    return param_groups


class EarlyStopping:
    def __init__(self, patience=2, delta=1e-3, path="best_model.pt"):
        """
        patience: How many epochs to wait after last time metric improved.
        delta: Minimum change in AUROC to qualify as an improvement.
        path: Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = -float('inf')
        self.early_stop = False

    def __call__(self, val_metric, model):
        # We want to MAXIMIZE metrics
        score = val_metric

        if score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        '''Saves model when validation metric increases.'''
        torch.save(model.state_dict(), self.path)