#!/usr/bin/env python3
"""
Multi-Modal Molecular Transformer (M3T) Demo
Executable version with simulated data for demonstration
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_simulated_data(n_samples=5000, n_mechanisms=100):
    """Generate simulated molecular data for demonstration."""
    
    print(f"Generating {n_samples} simulated molecular samples...")
    
    # Simulate SMILES-like sequences (simplified)
    vocab = ['C', 'N', 'O', 'S', 'P', '(', ')', '=', '#', '[', ']', '+', '-', '1', '2', '3', '4', '5', '6']
    vocab_with_special = ['<PAD>', '<UNK>', '<START>', '<END>'] + vocab
    char_to_idx = {char: idx for idx, char in enumerate(vocab_with_special)}
    
    # Generate random SMILES-like sequences
    max_length = 64
    smiles_sequences = []
    for _ in range(n_samples):
        seq_len = np.random.randint(10, max_length-4)
        sequence = [char_to_idx['<START>']]
        for _ in range(seq_len):
            sequence.append(np.random.choice(range(4, len(vocab_with_special))))
        sequence.append(char_to_idx['<END>'])
        
        # Pad sequence
        while len(sequence) < max_length:
            sequence.append(char_to_idx['<PAD>'])
        
        smiles_sequences.append(sequence[:max_length])
    
    tokenized_smiles = np.array(smiles_sequences)
    
    # Generate chemical descriptors (15 features)
    descriptors = np.random.randn(n_samples, 15)
    descriptors[:, 0] = np.random.uniform(100, 500, n_samples)  # Molecular weight
    descriptors[:, 1] = np.random.uniform(-2, 5, n_samples)    # LogP
    descriptors[:, 2] = np.random.randint(0, 10, n_samples)    # H-bond donors
    descriptors[:, 3] = np.random.randint(0, 15, n_samples)    # H-bond acceptors
    
    # Generate Morgan fingerprints (1024 bits)
    fingerprints = np.random.binomial(1, 0.1, (n_samples, 1024)).astype(np.float32)
    
    # Generate mechanism labels (multi-label)
    mechanism_names = [f"Mechanism_{i:03d}" for i in range(n_mechanisms)]
    labels = np.zeros((n_samples, n_mechanisms))
    
    # Each sample has 1-3 mechanisms
    for i in range(n_samples):
        n_labels = np.random.randint(1, 4)
        selected_mechanisms = np.random.choice(n_mechanisms, n_labels, replace=False)
        labels[i, selected_mechanisms] = 1
    
    return {
        'tokenized_smiles': tokenized_smiles,
        'descriptors': descriptors,
        'fingerprints': fingerprints,
        'labels': labels,
        'vocab': vocab_with_special,
        'char_to_idx': char_to_idx,
        'mechanism_names': mechanism_names
    }

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SMILESTransformerEncoder(nn.Module):
    """Transformer encoder for SMILES sequences."""
    
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, 
                 dim_feedforward=512, max_len=64, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # src shape: (batch_size, seq_len)
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        src = self.dropout(src)
        
        # Create padding mask
        if src_mask is None:
            src_mask = (src.sum(dim=-1) == 0)  # Padding positions
            
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        
        # Global average pooling (excluding padding)
        mask_expanded = (~src_mask).unsqueeze(-1).float()
        output_masked = output * mask_expanded
        output_pooled = output_masked.sum(dim=1) / mask_expanded.sum(dim=1)
        
        return output_pooled

class ChemicalDescriptorNetwork(nn.Module):
    """Neural network for processing chemical descriptors."""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusing different representations."""
    
    def __init__(self, seq_dim, desc_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.desc_proj = nn.Linear(desc_dim, hidden_dim)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, seq_features, desc_features):
        # Project to same dimension
        seq_proj = self.seq_proj(seq_features)  # (batch, hidden_dim)
        desc_proj = self.desc_proj(desc_features)  # (batch, hidden_dim)
        
        # Add sequence dimension for attention
        seq_proj = seq_proj.unsqueeze(1)  # (batch, 1, hidden_dim)
        desc_proj = desc_proj.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Cross attention: seq attends to desc
        attn_output, attn_weights = self.multihead_attn(
            query=seq_proj,
            key=desc_proj,
            value=desc_proj
        )
        
        # Residual connection and normalization
        output = self.norm1(seq_proj + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(output)
        output = self.norm2(output + ffn_output)
        
        return output.squeeze(1), attn_weights  # Remove sequence dimension

class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for molecular representation learning."""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, representations, fingerprints):
        """Compute contrastive loss between learned representations and fingerprints."""
        
        batch_size = representations.size(0)
        
        # Normalize representations
        repr_norm = F.normalize(representations, dim=1)
        fp_norm = F.normalize(fingerprints, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(repr_norm, fp_norm.T) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(batch_size).to(representations.device)
        
        # Compute InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class MultiModalMolecularTransformer(nn.Module):
    """Main M3T model combining all components."""
    
    def __init__(self, vocab_size, num_descriptors, num_classes, 
                 d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        
        # Component networks
        self.smiles_encoder = SMILESTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.descriptor_network = ChemicalDescriptorNetwork(
            input_dim=num_descriptors,
            hidden_dims=[256, 128, 64],
            dropout=dropout
        )
        
        self.cross_modal_attention = CrossModalAttention(
            seq_dim=d_model,
            desc_dim=64,
            hidden_dim=128,
            num_heads=nhead
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Contrastive learning head - match fingerprint dimension
        self.contrastive_head = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)  # Match fingerprint dimension
        )
        
    def forward(self, smiles_tokens, descriptors, return_attention=False):
        # Encode SMILES sequences
        seq_features = self.smiles_encoder(smiles_tokens)
        
        # Process chemical descriptors
        desc_features = self.descriptor_network(descriptors)
        
        # Cross-modal fusion
        fused_features, attention_weights = self.cross_modal_attention(
            seq_features, desc_features
        )
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Contrastive representation
        contrastive_repr = self.contrastive_head(fused_features)
        
        if return_attention:
            return logits, contrastive_repr, attention_weights
        else:
            return logits, contrastive_repr

class MolecularDataset(Dataset):
    """PyTorch dataset for multi-modal molecular data."""
    
    def __init__(self, smiles_tokens, descriptors, fingerprints, labels):
        self.smiles_tokens = torch.tensor(smiles_tokens, dtype=torch.long)
        self.descriptors = torch.tensor(descriptors, dtype=torch.float32)
        self.fingerprints = torch.tensor(fingerprints, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.smiles_tokens)
    
    def __getitem__(self, idx):
        return {
            'smiles_tokens': self.smiles_tokens[idx],
            'descriptors': self.descriptors[idx],
            'fingerprints': self.fingerprints[idx],
            'labels': self.labels[idx]
        }

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-MODAL MOLECULAR TRANSFORMER (M3T) - DEMO EXECUTION")
    print("=" * 80)
    
    # Generate simulated data
    data = generate_simulated_data(n_samples=2000, n_mechanisms=50)
    
    print(f"\nDataset Statistics:")
    print(f"Samples: {len(data['tokenized_smiles'])}")
    print(f"Vocabulary size: {len(data['vocab'])}")
    print(f"Sequence length: {data['tokenized_smiles'].shape[1]}")
    print(f"Descriptor features: {data['descriptors'].shape[1]}")
    print(f"Fingerprint bits: {data['fingerprints'].shape[1]}")
    print(f"Number of mechanisms: {len(data['mechanism_names'])}")
    
    # Normalize descriptors
    scaler = StandardScaler()
    descriptors_normalized = scaler.fit_transform(data['descriptors'])
    
    # Split data
    indices = np.arange(len(data['tokenized_smiles']))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.2, random_state=RANDOM_SEED)
    
    # Create datasets
    train_dataset = MolecularDataset(
        data['tokenized_smiles'][idx_train],
        descriptors_normalized[idx_train],
        data['fingerprints'][idx_train],
        data['labels'][idx_train]
    )
    
    val_dataset = MolecularDataset(
        data['tokenized_smiles'][idx_val],
        descriptors_normalized[idx_val],
        data['fingerprints'][idx_val],
        data['labels'][idx_val]
    )
    
    test_dataset = MolecularDataset(
        data['tokenized_smiles'][idx_test],
        descriptors_normalized[idx_test],
        data['fingerprints'][idx_test],
        data['labels'][idx_test]
    )
    
    # Create data loaders
    batch_size = 16  # Reduced for demo
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nData splits:")
    print(f"Training: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Initialize model
    model = MultiModalMolecularTransformer(
        vocab_size=len(data['vocab']),
        num_descriptors=descriptors_normalized.shape[1],
        num_classes=data['labels'].shape[1],
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nM3T Model Architecture:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_smiles = sample_batch['smiles_tokens'].to(device)
        sample_desc = sample_batch['descriptors'].to(device)
        
        logits, contrastive = model(sample_smiles, sample_desc)
        print(f"✓ Forward pass successful")
        print(f"Input SMILES shape: {sample_smiles.shape}")
        print(f"Input descriptors shape: {sample_desc.shape}")
        print(f"Output logits shape: {logits.shape}")
        print(f"Contrastive representation shape: {contrastive.shape}")
    
    print(f"\n" + "=" * 80)
    print("M3T DEMO SETUP COMPLETED SUCCESSFULLY!")
    print("Ready for training and evaluation...")
    print("=" * 80)
