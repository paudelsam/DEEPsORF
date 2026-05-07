from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score, 
                             f1_score, precision_score, recall_score, matthews_corrcoef,
                             roc_curve, confusion_matrix)
from sklearn.metrics import (RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay)

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from config import *

def plot_per_fold_loss(loss_file, save_path):
    """
    Plots train and validation loss for each folds
    
    :param loss_file: Path to loss file
    :param save_path: Path to save plot
    """
    df = pd.read_csv(loss_file, header=0, sep=",")

    fig, ax = plt.subplots(1, 2, figsize=(10,6), sharey=True)
    for fold in df["fold"].unique():
        fold_data = df[df["fold"] == fold]
        ax[0].plot(fold_data["epoch"], fold_data["train_loss"], label=f"Fold {fold}")
        ax[1].plot(fold_data["epoch"], fold_data["val_loss"])
        
    ax[0].set_title("Training loss")
    ax[1].set_title("Validation loss")
    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")
    ax[0].grid(True, alpha=0.3)
    ax[1].grid(True, alpha=0.3)
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_loss(loss_file, save_path):
    """
    Plots train and validation loss
    
    :param loss_file: Path to loss file
    :param save_path: Path to save plot
    """
    
    # Calculate average per group
    df = pd.read_csv(loss_file, header=0, sep=",")
    avg = df.groupby("epoch")[["train_loss", "val_loss"]].mean()
    std = df.groupby("epoch")[["train_loss", "val_loss"]].std()
    
    # Plot average train and validation loss 
    fig, ax = plt.subplots(1,1, figsize=(10,5))

    ax.plot(avg.index, avg["train_loss"], label="Train loss")
    ax.plot(avg.index, avg["val_loss"], label="Validation loss")

    ax.fill_between(avg.index, avg["train_loss"] - std["train_loss"], avg["train_loss"] + std["train_loss"], alpha=0.25,)
    ax.fill_between(avg.index, avg["val_loss"] - std["val_loss"], avg["val_loss"] + std["val_loss"], alpha=0.25)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Average train and validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    
# Calculate threshold of classification
def get_threshold_youden(y_true, y_prob):
    # Check for NaN or infinite values
    if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
        # Clamp probabilities to valid range
        y_prob = np.clip(y_prob, 1e-7, 1-1e-7)
        # Replace remaining NaN/Inf with 0.5 (neutral prediction)
        y_prob = np.where(np.isfinite(y_prob), y_prob, 0.5)
        
    fpr, tpr, thres = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thres[np.argmax(j)]) 


# Calculate evaluation metrics with NaN protection
def calculate_primary_metrics(y_true, y_prob, threshold):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
        # Clamp probabilities to valid range
        y_prob = np.clip(y_prob, 1e-7, 1-1e-7)
        # Replace remaining NaN/Inf with 0.5 (neutral prediction)
        y_prob = np.where(np.isfinite(y_prob), y_prob, 0.5)
    
    # Additional safety: clamp to valid probability range
    y_prob = np.clip(y_prob, 1e-7, 1-1e-7)
    
    try:
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "threshold": threshold
        }
    except Exception as e:
        # Return default metrics in case of error
        metrics = {
            "accuracy": 0.5, "roc_auc": 0.5, "pr_auc": 0.5,
            "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "mcc": 0.0, "threshold": 0.5
        }
    
    return metrics

def plot_roc_auc(y_true, y_score, save_path):
    auc = roc_auc_score(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(6,6))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Override sklearn defaults
    ax.set_ylabel("True positive rate")
    ax.set_xlabel("False positive rate")
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/roc_auc.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_pr_auc(y_true, y_score, save_path):
    aps = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6,6))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    
    baseline = float(np.mean(y_true))
    ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline = {baseline:.3f}')
    
    # Override sklearn defaults
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/pr_auc.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    

def plot_prediction_heatmap(y_true, y_prob, threshold, save_path):
    
    # Convert probabilities to predictions
    y_pred = (y_prob >= threshold).astype(int)
    labels = ["0", "1"]
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", 
                linewidths=1.0, linecolor="Black",
                cbar=False, cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/pred_heatmap.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()