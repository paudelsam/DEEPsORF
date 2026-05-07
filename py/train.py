# train.py - Train on the complete dataset

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path

from models import DEEPsORF

from dataset import ORFDataset

from config import *
from utils import *
from metrics import *

# Print info
print(f"Torch version: {torch.__version__}")
print(f"Cuda version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

# Disable cuDNN benchmarking
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Enable TF32 on Ampere GPUs for faster matrix multiplication
torch.set_float32_matmul_precision("high")

# Supress compilation warning
torch._dynamo.config.suppress_errors = True


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Load the dataset
    dataset = ORFDataset(TRAIN_ID_LABEL_FILE, TRAIN_EMB_FILE, debug_samples=None)
    
    TB_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=TB_DIR)
    
    # Create train and val loader
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_function,
                                num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    # Initialize the model, optimizer, and scheduler
    model = DEEPsORF().to(device)
    model.init_weights()
    
    param_groups = get_layerwise_optimizer_params(model, MAX_LR / DIV_FACTOR, WEIGHT_DECAY)
    optimizer = optim.AdamW(param_groups, lr=MAX_LR / DIV_FACTOR, fused=True)
    criterion = nn.BCEWithLogitsLoss()
    step_per_epoch = (len(train_loader) + ACCUMULATION_STEPS - 1) // ACCUMULATION_STEPS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                  max_lr=MAX_LR, 
                                                  epochs=EPOCHS, 
                                                  steps_per_epoch=step_per_epoch, 
                                                  pct_start=PCT_START,
                                                  anneal_strategy="cos",
                                                  div_factor=DIV_FACTOR, 
                                                  final_div_factor=FINAL_DIV_FACTOR)
    
    # Count the number of parameters and log them
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    writer.add_text("Total parameters", f"{total_params:,}", global_step=None)
    writer.add_text("Total trainable parameters", f"{trainable_params:,}", global_step=None)

    print(f"Total parameters: {total_params:,}")
    print(f"Total trainable parameters: {trainable_params:,}")
    
    global_step = 0    
    #------------------------------- EPOCH ----------------------------------
    for epoch in tqdm(range(1, EPOCHS+1), total=EPOCHS, desc=f"Training"):
        
        #------------------------------- TRAIN ---------------------------------- 
        model.train() 
        
        train_loss_sum = 0.0
        train_n = 0
    
        train_labels = []
        train_probs = []

        # Initialize gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        
        #------------------------------- BATCH ---------------------------------- 
        for batch_i, batch_data in enumerate(train_loader):
            embs = batch_data["embeddings"].to(device, non_blocking=True)
            masks = batch_data["mask"].to(device, non_blocking=True)
            labels = batch_data["labels"].to(device, non_blocking=True)
            
            smoothed = get_smoothed_labels(labels, EPSILON, CLASSES)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                
                logits, _, _, _ = model(embs, masks)
                loss = criterion(logits, smoothed)
                scaled_loss = loss / ACCUMULATION_STEPS         # Scaled loss for backpropagation
            
            scaled_loss.backward()
            
            # Update weights every accumulation_steps
            # Second condition is for last batch
            if (batch_i + 1) % ACCUMULATION_STEPS == 0 or ((batch_i + 1) == len(train_loader)):
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
                optimizer.step()
                scheduler.step()
                
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("GradNorm", total_norm, global_step)
                global_step += 1
                
                optimizer.zero_grad(set_to_none=True)
                
            # Accumulate batch loss (bs = batch size)
            bs = labels.size(0)
            train_loss_sum += loss.item() * bs 
            train_n += bs
            
            # Accumulate train prediction 
            probs = torch.sigmoid(torch.clamp(logits, -10, 10)).detach().float().cpu().numpy()
            train_labels.append(labels.cpu().numpy())
            train_probs.append(probs)
        
        train_loss_epoch = train_loss_sum / train_n  
        
        train_labels_concat = np.concatenate(train_labels)
        train_probs_concat = np.concatenate(train_probs)
            
        # Fixed 0.5 threshold for training set
        train_preds_concat = (train_probs_concat >= 0.5).astype(int)
        train_acc_epoch = accuracy_score(train_labels_concat, train_preds_concat)
            
        # TensorBoard logging
        writer.add_scalar(f"Train loss", train_loss_epoch, epoch)
        writer.add_scalar(f"Train accuracy", train_acc_epoch, epoch)

    # Save the model
    print(f"Saving model to {FINAL_MODEL_DIR}")
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{FINAL_MODEL_DIR}/DEEPsORF.pt")
        
    # Close summary writer
    writer.close()       

if __name__ == "__main__":
    main()     