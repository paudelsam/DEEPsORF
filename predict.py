# Import required packages

import os
from pathlib import Path
import numpy as np
import h5py
import argparse
import pandas as pd
from tqdm import tqdm


import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel, AutoConfig

import sys 
sys.path.append("./py")
from models import DEEPsORF

class SeqDataset(Dataset):
    """For embedding extraction - reads sequences from TSV"""
    def __init__(self, tsv_file):
        self.df = pd.read_csv(tsv_file, sep="\t")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["id"], row["strand"], row["start"], row["end"], row["orf"]


def print_banner():
    logo = r"""                                                                                      
 ▄▄▄▄   ▄▄▄▄▄▄ ▄▄▄▄▄▄ ▄▄▄▄▄          ▄▄▄▄  ▄▄▄▄▄  ▄▄▄▄▄▄
 █   ▀▄ █      █      █   ▀█  ▄▄▄   ▄▀  ▀▄ █   ▀█ █     
 █    █ █▄▄▄▄▄ █▄▄▄▄▄ █▄▄▄█▀ █   ▀  █    █ █▄▄▄▄▀ █▄▄▄▄▄
 █    █ █      █      █       ▀▀▀▄  █    █ █   ▀▄ █     
 █▄▄▄▀  █▄▄▄▄▄ █▄▄▄▄▄ █      ▀▄▄▄▀   █▄▄█  █    ▀ █                                                                                                
"""
    print(logo)
    print("="*60)
    print("Prediction pipeline for human sORFs")
    print("Author: Samrat Paudel")
    print("Version: 1.0")
    print("="*60)
    print("\n")
                                

def get_args():
    parser = argparse.ArgumentParser(
        description="Run DEEPsORF model"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--seq_file", type=str, required=True, help="Path to sequence file (.fasta or .tsv)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for batch processing")
    parser.add_argument("--output", type=str, default="./output", help="Path for output files")
    parser.add_argument("--input_orf", action="store_true", help="Input sequence is ORF. Skips ORF prediction step")
    parser.add_argument("--attn_out", action="store_true", help="Weather to output attention scores")
    
    
    return parser.parse_args()
        
    
def read_file(file):
    # Can read both .fasta and .tsv
    extension = Path(file).suffix
    
    if extension == ".fasta":
        print(f"\nFile format detected: 'fasta'")
        with open(file) as f:
            seq_id = None
            seq_lines = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq_id:
                        yield seq_id, "".join(seq_lines)
                    seq_id = line[1:].split()[0]
                    seq_lines = []
                else:
                    seq_lines.append(line)
            if seq_id:
                yield seq_id, "".join(seq_lines)
                
    if extension == ".tsv":
        print(f"\nFile format detected: 'tsv'")
        df = pd.read_csv(file, sep="\t", header=None)
        for _, row in df.iterrows():
            id = row[0]
            seq = row[1]
            yield id, seq
 
            
def reverse_complement(seq):
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]

def find_orfs(seq, 
              strand="+", 
              start_codons=("ATG", "CTG", "GTG", "AAG", "AGG", "ATC", "TTG", "ATT", "ACG", "ATA"),
              stop_codons=("TAA", "TAG", "TGA"),
              min_len=30, 
              max_len=300):
    seq = seq.upper()
    if strand == "-":
        seq = reverse_complement(seq)

    orfs = []
    seq_len = len(seq)
    
    for frame in range(3):
        starts = []  # track all start positions in this frame
        
        for i in range(frame, seq_len - 2, 3):
            codon = seq[i:i+3]
            
            if codon in start_codons:
                starts.append(i)
            
            elif codon in stop_codons:
                # pair this stop with ALL pending starts
                for start_pos in starts:
                    orf_seq = seq[start_pos:i+3]
                    if min_len <= len(orf_seq) <= max_len:
                        if strand == "+":
                            s, e = start_pos + 1, i + 3
                        else:
                            s, e = seq_len - (i + 3) + 1, seq_len - start_pos
                        orfs.append((strand, s, e, orf_seq))
                starts = []  # reset after stop codon
    
    return orfs


def extract_orfs_to_tsv(file, output):
    orf_file = os.path.join(output, "orfs.tsv")
    orf_list = []

    for seq_id, sequence in read_file(file):
        # forward strand
        for strand, start, end, orf_seq in find_orfs(sequence, strand="+"):
            orf_id = f"{seq_id}_PLUS_{start}_{end}"
            orf_list.append({
                "id": orf_id, "parent_id": seq_id, "strand": strand,
                "start": start, "end": end, "orf": orf_seq
            })
        # reverse strand
        for strand, start, end, orf_seq in find_orfs(sequence, strand="-"):
            orf_id = f"{seq_id}_MINUS_{start}_{end}"
            orf_list.append({
                "id": orf_id, "parent_id": seq_id, "strand": strand,
                "start": start, "end": end, "orf": orf_seq
            })

    df_orfs = pd.DataFrame(orf_list)
    df_orfs.to_csv(orf_file, sep="\t", index=False)
    print(f"ORFs extracted and saved.")
    return orf_file

def reformat_to_tsv(file, output):
    """
    I am converting input file to ORF file format without extracting orfs
    This function follows naming convention of extract_orfs_to_tsv
    """
    
    orf_file = os.path.join(output, "orfs.tsv") 
    orf_list = []
    
    for seq_id, sequence in read_file(file):
        orf_list.append({
            "id": seq_id, "parent_id": seq_id, "strand": ".",
            "start": 1, "end": len(sequence), "orf": sequence 
        })
    
    df_orfs = pd.DataFrame(orf_list)
    df_orfs.to_csv(orf_file, sep="\t", index=False)
    print(f"File reformatted and saved.")
    return orf_file   
    
def get_embeddings(ids, orfs, model, tokenizer, device):
    model.eval()
    
    # Embeddings extraction loop    
    inputs = tokenizer(list(orfs), return_tensors="pt", padding=True, 
                        truncation=True, max_length=512, return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")
    inputs = inputs.to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
            embeddings = outputs[0].cpu().numpy()       
            
    # Save each seq in the batch individually 
    emb_dict = {}
    for i, id in enumerate(ids):
        # Since we padded sequence for batch operation
        # Remove PAD token before saving and [CLS] as well
        # In [1, 1:actual_len-1, :], Keep B, remove [CLS] [PAD] [SEP], keep 768-D
        actual_len = inputs["attention_mask"][i].sum().item()
        actual_emb = embeddings[i, 1:actual_len-1, :]
        actual_offsets = offsets[i, 1:actual_len-1, :].numpy()
        
        emb_dict[id] = {"emb": actual_emb,
                        "offsets": actual_offsets} 
        
    # Clean up
    del inputs, outputs, embeddings, offsets
    return emb_dict
    
# Map token level attention to base level attention
def token_to_base_attention(token_attn, offsets, seq_len):
    base_attn = np.zeros(seq_len)
    if hasattr(offsets, "numpy"):
               offsets = offsets.numpy()
    if hasattr(token_attn, "numpy"):
        token_attn = token_attn.numpy()
    
    for i, (start, end) in enumerate(offsets):
        start, end = int(start), int(end)
        if start == end:
            continue
        base_attn[start:end] = token_attn[i]
    return base_attn
    

          
def load_model(model_path, device):
    model = DEEPsORF()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    print(f"\nModel loaded in evaluation mode.")
    
    return model


def main():
    print_banner()
    
    # Get args
    args = get_args()

    # Create output dir 
    os.makedirs(args.output, exist_ok=True)
    
    # Clear old files if present
    old_files = ["orfs.tsv", "prediction.tsv", "attention.h5"]
    for file in old_files:
        path = Path(args.output) / file
        if path.exists():
            try:
                path.unlink(missing_ok=True)
                print(f"Removed existing files: {file}")
            except OSError as e:
                print("Error removing existing file(s)")
                print("You might need to manually remove them before running prediction.")
    
    if args.input_orf: 
        # No need to obtain ORFs. Only formatting.
        orf_file = reformat_to_tsv(args.seq_file, args.output)  
        
    else: 
        # Obtain ORFs
        orf_file = extract_orfs_to_tsv(args.seq_file, args.output)
    
    # Configurations for DNABERT2 model
    cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load tokenizer and model
    print(f"Loading tokenizer and model")

    config = AutoConfig.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    config.pad_token_id = 0


    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", 
        trust_remote_code=True, 
        cache_dir=cache_dir
        )

    dnabert_model = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        config=config,
        trust_remote_code=True,
        cache_dir=cache_dir
        )

    # Disable Triton flash attention — incompatible API in newer Triton versions.
    import sys
    for mod_name, mod in sys.modules.items():
        if "bert_layers" in mod_name and "zhihan1996" in mod_name:
            mod.flash_attn_qkvpacked_func = None
            break

    dnabert_model = dnabert_model.to(device)
    
    # Load trained model
    DEEPsORF_model = load_model(args.model, device)
    
    # Create batch of sequences
    dataset = SeqDataset(orf_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Text file for prediction output
    out_file = f"{args.output}/prediction.tsv"
    
    # Open file once outside batch loop
    with open(out_file, "w", newline="") as out:
        out.write(f"id\tstrand\tstart\tend\tpreds\n")
    
        # Operate on each batch, pass batch of sequences
        for batch_id, batch in enumerate(tqdm(dataloader, desc="Embedding extraction and prediction")):
            
            # Obtain embeddings
            ids, strands, starts, ends, orfs = batch
            emb_dict = get_embeddings(ids, orfs, dnabert_model, tokenizer, device)
            
            # Collect embs from emb_dict for prediction
            embs = [torch.tensor(emb_dict[id]["emb"], dtype=torch.float32) for id in ids]
            offsets = [emb_dict[id]["offsets"] for id in ids]
            
            # Pad sequence and create mask
            lengths = torch.tensor([e.shape[0] for e in embs])
            emb_padded = pad_sequence(embs, batch_first=True)
            max_len = emb_padded.size(1)
            mask = (torch.arange(max_len)[None, :] < lengths[:, None]).unsqueeze(1)
            
            # Run predictions
            emb_padded = emb_padded.to(device)
            mask = mask.to(device)
            
            with torch.no_grad():
                if args.attn_out:
                    logits, cnn_attn5, cnn_attn3, bigru_attn, = DEEPsORF_model(emb_padded, mask)
                    preds = torch.sigmoid(torch.clamp(logits, -10, 10)).cpu().numpy()
                    
                    cnn5 = cnn_attn5.cpu().numpy()
                    cnn3 = cnn_attn3.cpu().numpy()  
                    bigru = bigru_attn.cpu().numpy()
                    
                    with h5py.File(f"{args.output}/attention.h5", "a") as f:
                        for i, sid in enumerate(ids):
                            seq_len = ends[i] - starts[i] + 1
                            offsets_i = offsets[i]
                            
                            # Map each attention to base level
                            cnn5_base = token_to_base_attention(cnn5[i], offsets_i, seq_len)
                            cnn3_base = token_to_base_attention(cnn3[i], offsets_i, seq_len)
                            bigru_base = token_to_base_attention(bigru[i], offsets_i, seq_len)
                            
                            out.write(f"\n{sid}\t{strands[i]}\t{starts[i]}\t{ends[i]}\t{preds[i]:.4f}")
                            
                            # Create group for this seq
                            g = f.require_group(sid)
                            
                            # Save attention array
                            for name, data in {"cnn5": cnn5_base, "cnn3": cnn3_base, "bigru": bigru_base}.items():
                                if name in g:
                                    del g[name]
                                g.create_dataset(name, data=data, compression="gzip")
                    
                else:
                    logits, _, _, _ = DEEPsORF_model(emb_padded, mask)
                    preds = torch.sigmoid(torch.clamp(logits, -10, 10)).cpu().numpy()
                    for i, sid in enumerate(ids):
                        out.write(f"{sid}\t{strands[i]}\t{starts[i]}\t{ends[i]}\t{preds[i]:.4f}\t{orfs[i]}\n")
    out.close()
    
    print("Prediction completed!")
    print(f"Prediction file saved: {out_file}")
    if args.attn_out:
        print(f"Attention scores saved: {args.output}/attention.h5")  
          
if __name__ == "__main__":
    main()