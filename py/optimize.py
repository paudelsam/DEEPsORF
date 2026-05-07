from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import StratifiedKFold


from models import DEEPsORF
from dataset import ORFDataset

from config import *
from utils import *
from metrics import *

# Print info
print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
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

    # Create folds
    k_folds = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    
    # -------------------------- FOLDS -----------------------------------
    for fold, (train_idx, val_idx) in enumerate(k_folds.split(X=dataset.id_label, y=dataset.id_label["label"])):
        TB_DIR.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=f"{TB_DIR}/fold_{fold}")
        
        # Create train and validation subset
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        
        # Create train and val loader
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_function,
                                  num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        
        val_loader = DataLoader(val_set, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_function,
                                num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        
        # Initialize the model, optimizer, and scheduler
        model = DEEPsORF().to(device)
        model.init_weights()
        #model = torch.compile(model, mode="reduce-overhead")
        
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
        
        
        checkpoints = CHECKPOINTS_DIR / f"fold_{fold}.pt"
        early_stopping = EarlyStopping(PATIENCE, DELTA, str(checkpoints))

        fold_loss_train = []
        fold_loss_val = []
        
        global_step = 0
        
        #------------------------------- EPOCH ----------------------------------
        for epoch in tqdm(range(1, EPOCHS+1), total=EPOCHS, desc=f"Training on {fold} of {K_FOLDS}"):
            
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
                    # logits, cnn_attn, bigru_attn, We don't need attn here
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
                
            #------------------------------- EVAL ---------------------------------- 
            model.eval()
            
            val_labels = []
            val_probs = []
            
            val_loss_sum = 0.0
            val_n = 0
            
            with torch.no_grad():   
                for batch_data in val_loader: 
                    embs = batch_data["embeddings"].to(device, non_blocking=True)
                    masks = batch_data["mask"].to(device, non_blocking=True)
                    labels = batch_data["labels"].to(device, non_blocking=True)
                    
                    logits, _, _, _ = model(embs, masks)
                    
                    if torch.isnan(logits).any():
                        print(f"NaN detected in logits!")
                        print(f"Input has NaN: {torch.isnan(embs).any()}")
                        print(f"Mask all False: {(~masks).all(dim=-1).any()}")
    
                    loss = criterion(logits, labels.float())
                        
                    bs = labels.size(0)
                    val_loss_sum += loss.item() * bs
                    val_n += bs
                    
                    # Move to CPU
                    probs = torch.sigmoid(torch.clamp(logits, -10, 10)).detach().float().cpu().numpy()
                    val_labels.append(labels.cpu().numpy())
                    val_probs.append(probs)
                        
            val_loss_epoch = val_loss_sum / val_n
            
            # Calculate metrics on each epoch
            val_labels_concat = np.concatenate(val_labels)
            val_probs_concat = np.concatenate(val_probs)
            
            train_labels_concat = np.concatenate(train_labels)
            train_probs_concat = np.concatenate(train_probs)
            
            # Fixed 0.5 threshold for training set
            train_preds_concat = (train_probs_concat >= 0.5).astype(int)
            train_acc_epoch = accuracy_score(train_labels_concat, train_preds_concat)
            
            # Calculated threshold for validation set
            threshold = get_threshold_youden(val_labels_concat, val_probs_concat)
            val_preds_concat = (val_probs_concat >= threshold).astype(int)
            val_acc_epoch = accuracy_score(val_labels_concat, val_preds_concat)
            
            # Split scores by class for histogram
            pos_score = val_probs_concat[val_labels_concat == 1]
            neg_score = val_probs_concat[val_labels_concat == 0]
            
            # Update the loss to fold loss list
            fold_loss_train.append(train_loss_epoch)
            fold_loss_val.append(val_loss_epoch)

            # Calculate metrics and TensorBoard logging
            metrics = calculate_primary_metrics(val_labels_concat, val_probs_concat, threshold)
            writer.add_scalar(f"Threshold", threshold, epoch)
            writer.add_scalar(f"Train loss", train_loss_epoch, epoch) 
            writer.add_scalar(f"Val loss", val_loss_epoch, epoch)
            writer.add_scalar(f"Train accuracy", train_acc_epoch, epoch)
            writer.add_scalar(f"Val accuracy", val_acc_epoch, epoch)
            writer.add_scalar(f"F1", metrics["f1"], epoch)
            writer.add_scalar(f"Precision", metrics["precision"], epoch)
            writer.add_scalar(f"Recall", metrics["recall"], epoch)
            writer.add_scalar(f"AUC_ROC", metrics["roc_auc"], epoch)
            writer.add_scalar(f"AUPRC", metrics["pr_auc"], epoch)
            writer.add_scalar(f"MCC", metrics["mcc"], epoch)
            
            # Only add histograms if arrays are not empty and contain valid values
            if len(pos_score) > 0 and np.isfinite(pos_score).any() and len(np.unique(pos_score[np.isfinite(pos_score)])) > 1:
                writer.add_histogram("Val positive scores", pos_score[np.isfinite(pos_score)], epoch)
                writer.add_scalar("Val pos score: NA", np.isnan(pos_score).sum(), epoch)
                writer.add_scalar("Val pos score: INF", np.isinf(pos_score).sum(), epoch)

            if len(neg_score) > 0 and np.isfinite(neg_score).any() and len(np.unique(neg_score[np.isfinite(neg_score)])) > 1:
                writer.add_histogram("Val negative scores", neg_score[np.isfinite(neg_score)], epoch)
                writer.add_scalar("Val neg score: NA", np.isnan(neg_score).sum(), epoch)
                writer.add_scalar("Val neg score: INF", np.isinf(neg_score).sum(), epoch)
                
            # Early stopping
            early_stopping(metrics["f1"], model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at {epoch}.")
                break
            
        # Write the per fold loss to csv
        mode = "w" if fold == 0 else "a"
        with open(LOSS_FILE, mode, newline="") as f:
            if fold == 0:
                f.write("fold,epoch,train_loss,val_loss\n")
            for idx, (item1, item2) in enumerate(zip(fold_loss_train, fold_loss_val)):
                f.write(f"{fold},{idx},{item1},{item2}\n")  
    
        # Clear CUDA cache between folds
        del model, optimizer, scheduler
            
        # Close summary writer
        writer.close()   
        
    print("Training completed!!!")
    plot_per_fold_loss(LOSS_FILE, f"{RESULT_DIR}/per_fold_loss.png")  
    plot_loss(LOSS_FILE, f"{RESULT_DIR}/loss.png")  

if __name__ == "__main__":
    main()