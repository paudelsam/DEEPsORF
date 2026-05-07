import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttentionPool(nn.Module):
    def __init__(self, input_emb, heads=4, attn_dropout=0.10, temp=0.3):
        super().__init__()
        self.heads = heads
        self.attn_logits = nn.Conv1d(input_emb, heads, kernel_size=1, bias=True)
        self.attn_dropout = attn_dropout
        self.temperature = temp
        self.layer_norm = nn.LayerNorm(input_emb)
        
        # Project concatenated multi-head output back to input_emb dim
        self.output_proj = nn.Linear(input_emb * heads, input_emb)
       
    def forward(self, x, mask):
        # x shape: [Batch, Emb, Length]
        # mask shape: [Batch, 1, Length]
        x_norm = self.layer_norm(x.transpose(1,2)).transpose(1,2)
        logits = self.attn_logits(x_norm)             # [B, heads, L]
        logits = logits.masked_fill(~mask, -1e9)
        
        w = torch.softmax(logits / self.temperature, dim=-1)     # [B, heads, L]
        
        # Dropout across positions
        if self.attn_dropout > 0.0:
            w = F.dropout(w, p=self.attn_dropout, training=self.training)
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Pooled per head   
        pooled = torch.einsum("bel,bhl->bhe", x_norm, w)
        
        # Reshape to [B, heads*emb] and project
        pooled = pooled.reshape(pooled.size(0), -1)
        pooled = self.output_proj(pooled)    #[B, emb]
        
        # Average attention for visualization
        w_mean = w.mean(dim=1)
        
        return pooled, w_mean


class CNNWithAttention(nn.Module):
    def __init__(self, input_emb=768, emb_dropout=0.1, conv_out=128):
        super().__init__()
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.identity_proj = nn.Conv1d(input_emb, conv_out, kernel_size=1)
        
        # Build convolutional layers. Conv branch with kernel_size = 5
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_emb, 256, kernel_size=5, padding=5//2, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout1d(0.3),
            nn.Conv1d(256, conv_out, kernel_size=5, padding=5//2, bias=False),
            nn.BatchNorm1d(conv_out),
            nn.GELU(),
            nn.Dropout1d(0.3)
        )
        
        # Second conv branch with kernel_size = 3
        self.conv2 = nn.Sequential(
            nn.Conv1d(input_emb, 256, kernel_size=3, padding=3//2, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout1d(0.3),
            nn.Conv1d(256, conv_out, kernel_size=3, padding=3//2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout1d(0.3)
        )
        
        # Attention pooling
        self.attention_pool = MultiHeadAttentionPool(conv_out)
            
    def forward(self, x, mask):
        # x = [B, L, C]
        x = self.emb_dropout(x)
        x = x.transpose(1, 2)
        
        identity = self.identity_proj(x)
        x1 = self.conv1(x) + identity
        x2 = self.conv2(x) + identity
        
        pooled1, attn_weights1 = self.attention_pool(x1, mask=mask)
        pooled2, attn_weights2 = self.attention_pool(x2, mask=mask)
    
        return pooled1, pooled2, attn_weights1, attn_weights2



class BIGRU(nn.Module):
    def __init__(self, input_size=768, gru_out=128, num_layers=2):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_out,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, 
            dropout = 0.05 if num_layers > 1 else 0.0)
    
        # Attention pooling over GRU outputs
        self.attention_pool = MultiHeadAttentionPool(gru_out*2, heads=2, attn_dropout=0.15, temp=0.8)    
    
    def forward(self, x, mask):
        output, _ = self.bigru(x)   #[B, L, gru_out*2]
        
        # Multihead attention expects [B, Emb, L]
        output = output.transpose(1,2)
        attended, attn_weights = self.attention_pool(output, mask)
        return attended, attn_weights
    
    
class DEEPsORF(nn.Module):
    def __init__(self, input_emb=768, conv_out=128, gru_out=256, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        
        # Two branches for CNN and BiGRU
        self.cnn_branch = CNNWithAttention(input_emb=input_emb)
        self.bigru_branch = BIGRU(input_size=input_emb)
        
        # Inputs from CNN and BiGRU
        mlp_input = conv_out + conv_out + gru_out
        
        # Multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    # Initialize weights
    # Using weight inits supposed to be better than default init
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)  
          
               
    def forward(self, x, mask):
        cnn1_out, cnn2_out, cnn_attn_weight1, cnn_attn_weight2 = self.cnn_branch(x, mask)
        bigru_out, bigru_attn_weight = self.bigru_branch(x, mask)

        # Concat features
        combined_features = torch.cat([cnn1_out, cnn2_out, bigru_out], dim=-1)

        # Pass through MLP for logits
        logits = self.mlp(combined_features)
        
        return logits.squeeze(-1), cnn_attn_weight1, cnn_attn_weight2, bigru_attn_weight
       