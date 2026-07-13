import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusion(nn.Module):
    def __init__(self, v_dim, t_dim, num_classes, strategy='late_fusion', alpha=0.5):
        super(MultimodalFusion, self).__init__()
        self.strategy = strategy
        self.num_classes = num_classes
        self.alpha = alpha
        
        # 1. Concatenation + MLP Strategy
        if self.strategy == 'concat_mlp':
            self.mlp = nn.Sequential(
                nn.Linear(v_dim + t_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        # 2. Cross Attention Strategy
        elif self.strategy == 'cross_attention':
            # Project both modalities to a common embedding dimension
            self.embed_dim = min(v_dim, t_dim) 
            self.v_proj = nn.Linear(v_dim, self.embed_dim)
            self.t_proj = nn.Linear(t_dim, self.embed_dim)
            
            # Cross-attention layer (Text queries Vision, or vice versa)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim, 
                num_heads=4, 
                batch_first=True,
                dropout=0.2
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
            
        # 3. Late Fusion (Weighted Logits)
        elif self.strategy == 'late_fusion':
            self.v_classifier = nn.Linear(v_dim, num_classes)
            self.t_classifier = nn.Linear(t_dim, num_classes)

    def forward(self, v_features, t_features):
        """
        v_features: Tensor of shape (Batch, v_dim)
        t_features: Tensor of shape (Batch, t_dim)
        """
        if self.strategy == 'concat_mlp':
            # Concatenate along the feature dimension
            fused = torch.cat((v_features, t_features), dim=-1)
            logits = self.mlp(fused)
            return logits
            
        elif self.strategy == 'cross_attention':
            # Project to common dimension and add sequence length dimension -> (Batch, Seq=1, Dim)
            v_proj = self.v_proj(v_features).unsqueeze(1) 
            t_proj = self.t_proj(t_features).unsqueeze(1) 
            
            # Cross Attention: Text features act as Query, Vision features as Key & Value
            # (Allows text to find relevant visual features)
            attn_output, _ = self.cross_attn(query=t_proj, key=v_proj, value=v_proj)
            
            # Remove sequence dimension and classify -> (Batch, Dim)
            attn_output = attn_output.squeeze(1)
            logits = self.classifier(attn_output)
            return logits
            
        elif self.strategy == 'late_fusion':
            # Classify separately, then fuse probabilities
            v_logits = self.v_classifier(v_features)
            t_logits = self.t_classifier(t_features)
            
            v_probs = F.softmax(v_logits, dim=-1)
            t_probs = F.softmax(t_logits, dim=-1)
            
            # Weighted average
            fused_probs = (self.alpha * t_probs) + ((1 - self.alpha) * v_probs)
            # Log(probs) to simulate logits for CrossEntropyLoss during training
            return torch.log(fused_probs + 1e-8)