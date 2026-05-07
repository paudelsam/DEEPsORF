from tqdm import tqdm

import torch
import h5py
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import DEEPsORF
from dataset import ORFDataset

from config import *
from utils import *
from metrics import *

# Enable TF32 on Ampere GPUs for faster matrix multiplication
torch.set_float32_matmul_precision("high")

# Supress compilation warning
torch._dynamo.config.suppress_errors = True

def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    writer = SummaryWriter(log_dir=TB_DIR)
    
    # Load the dataset
    dataset = ORFDataset(TEST_ID_LABEL_FILE, TEST_EMB_FILE, debug_samples=None)
    writer.add_text("Total samples", str(len(dataset.id_label)))
    
    # Create test loader
    test_loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_function,
                                num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    # Load the saved model from training
    best_model_path = FINAL_MODEL
    
    print(f"Loading trained model from:\n\tPath:{best_model_path}")
    model = DEEPsORF().to(device)
    model.load_state_dict(torch.load(best_model_path))

    # Set the model to evaluation mode
    model.eval()
    
    test_labels = []
    test_probs = []
    
    with torch.no_grad():        
        for batch_data in test_loader: 
            embs = batch_data["embeddings"].to(device, non_blocking=True)
            masks = batch_data["mask"].to(device, non_blocking=True)
            labels = batch_data["labels"].to(device, non_blocking=True)
            
            logits, _, _, _ = model(embs, masks)
            
            # Accumulate test predictions
            probs = torch.sigmoid(torch.clamp(logits, -10, 10)).detach().float().cpu().numpy()
            test_labels.append(labels.cpu().numpy())
            test_probs.append(probs)
        
    test_labels_concat = np.concatenate(test_labels)
    test_probs_concat = np.concatenate(test_probs)
    
    # Threshold from validation set
    threshold = 0.4469
    test_preds_concat = (test_probs_concat >= threshold).astype(int)
    
    # Split scores by class for histogram
    pos_score = test_preds_concat[test_labels_concat == 1]
    neg_score = test_preds_concat[test_labels_concat == 0]
    
    # Calculate metrics and log to tensorboard
    metrics = calculate_primary_metrics(test_labels_concat, test_probs_concat, threshold)
    summary_text = (
    "## Test Results\n"
    f"- **Accuracy**: {metrics['accuracy']:.4f}\n"
    f"- **Precision**: {metrics['precision']:.4f}\n"
    f"- **Recall**: {metrics['recall']:.4f}\n"
    f"- **F1**: {metrics['f1']:.4f}\n"
    f"- **AUC-ROC**: {metrics['roc_auc']:.4f}\n"
    f"- **PR-AUC**: {metrics['pr_auc']:.4f}\n"
    f"- **MCC**: {metrics['mcc']:.4f}\n"
)
    writer.add_text("Test Summary", summary_text, 0)
    
    writer.add_histogram("Val positive scores", pos_score, 0)
    writer.add_histogram("Val negative score", neg_score, 0)
    
    print(summary_text)

    # Close summary writer
    writer.close() 
    
    # Save actuals and predictions
    actuals_preds = np.column_stack((test_labels_concat, test_probs_concat))
    np.savetxt(f"{RESULT_DIR}/actuals_prediction.txt", actuals_preds, delimiter=",")
    
    # Plot curves
    plot_roc_auc(test_labels_concat, test_probs_concat, f"{RESULT_DIR}")
    plot_pr_auc(test_labels_concat, test_probs_concat, f"{RESULT_DIR}")
    plot_prediction_heatmap(test_labels_concat, test_probs_concat, threshold, f"{RESULT_DIR}")
    

if __name__ == "__main__":
    main()
    
                
               
                
                
                    
            
            
        
        
        
        