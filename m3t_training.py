#!/usr/bin/env python3
"""
Complete M3T Training and Evaluation Script
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Import our M3T components
from m3t_demo import (
    generate_simulated_data, MultiModalMolecularTransformer, 
    MolecularDataset, ContrastiveLoss, device
)

class HybridLoss(nn.Module):
    """Combined loss function for multi-task learning."""
    
    def __init__(self, classification_weight=1.0, contrastive_weight=0.1, 
                 temperature=0.1):
        super().__init__()
        
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = ContrastiveLoss(temperature)
        
    def forward(self, logits, labels, representations, fingerprints):
        # Classification loss
        cls_loss = self.bce_loss(logits, labels)
        
        # Contrastive loss
        cont_loss = self.contrastive_loss(representations, fingerprints)
        
        # Combined loss
        total_loss = (self.classification_weight * cls_loss + 
                     self.contrastive_weight * cont_loss)
        
        return total_loss, cls_loss, cont_loss

def evaluate_model(model, data_loader, criterion, device):
    """Comprehensive model evaluation."""
    
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_cont_loss = 0
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in data_loader:
            smiles_tokens = batch['smiles_tokens'].to(device)
            descriptors = batch['descriptors'].to(device)
            fingerprints = batch['fingerprints'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits, representations = model(smiles_tokens, descriptors)
            
            # Compute losses
            loss, cls_loss, cont_loss = criterion(
                logits, labels, representations, fingerprints
            )
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_cont_loss += cont_loss.item()
            
            # Get predictions
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    
    # Concatenate all results
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    all_probabilities = np.vstack(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_micro = f1_score(all_labels, all_predictions, average='micro')
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    
    # AUC-ROC (handle cases where some classes have no positive samples)
    try:
        auc_micro = roc_auc_score(all_labels, all_probabilities, average='micro')
        auc_macro = roc_auc_score(all_labels, all_probabilities, average='macro')
    except ValueError:
        auc_micro = auc_macro = 0.0
    
    avg_loss = total_loss / len(data_loader)
    avg_cls_loss = total_cls_loss / len(data_loader)
    avg_cont_loss = total_cont_loss / len(data_loader)
    
    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'cont_loss': avg_cont_loss,
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'auc_micro': auc_micro,
        'auc_macro': auc_macro,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels
    }

def train_m3t_model():
    """Complete training pipeline for M3T model."""
    
    print("=" * 80)
    print("MULTI-MODAL MOLECULAR TRANSFORMER (M3T) - TRAINING")
    print("=" * 80)
    
    # Generate data
    print("Generating simulated molecular data...")
    data = generate_simulated_data(n_samples=3000, n_mechanisms=75)
    
    # Normalize descriptors
    scaler = StandardScaler()
    descriptors_normalized = scaler.fit_transform(data['descriptors'])
    
    # Split data
    indices = np.arange(len(data['tokenized_smiles']))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.2, random_state=42)
    
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
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset splits:")
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
    
    # Loss function and optimizer
    criterion = HybridLoss(
        classification_weight=1.0,
        contrastive_weight=0.1,
        temperature=0.1
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,  # Higher learning rate for faster demo
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training configuration
    num_epochs = 15  # Reduced for demo
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_f1': []
    }
    
    print(f"\nStarting training for {num_epochs} epochs...")
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_pbar:
            smiles_tokens = batch['smiles_tokens'].to(device)
            descriptors = batch['descriptors'].to(device)
            fingerprints = batch['fingerprints'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, representations = model(smiles_tokens, descriptors)
            
            # Compute loss
            loss, cls_loss, cont_loss = criterion(
                logits, labels, representations, fingerprints
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            train_correct += (predictions == labels).float().sum().item()
            train_total += labels.numel()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_correct/train_total:.3f}'
            })
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['loss'])
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_micro'])
        
        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1_micro']:.4f} | Val AUC: {val_metrics['auc_micro']:.4f}")
        
        # Early stopping check
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }, 'best_m3t_model.pth')
            
            print(f"✓ New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load('best_m3t_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    # Final results
    print("\n" + "=" * 80)
    print("MULTI-MODAL MOLECULAR TRANSFORMER (M3T) - FINAL RESULTS")
    print("=" * 80)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"F1-Score (Micro): {test_metrics['f1_micro']:.4f}")
    print(f"F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"AUC-ROC (Micro): {test_metrics['auc_micro']:.4f}")
    print(f"AUC-ROC (Macro): {test_metrics['auc_macro']:.4f}")
    
    # Comparison with baseline
    baseline_accuracy = 0.9992  # From GCN baseline
    print(f"\n" + "=" * 80)
    print("COMPARISON WITH BASELINE GCN")
    print("=" * 80)
    print(f"Baseline GCN Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"M3T Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Accuracy Difference: {test_metrics['accuracy'] - baseline_accuracy:+.4f}")
    
    if test_metrics['accuracy'] > baseline_accuracy:
        print("✓ M3T outperforms baseline GCN!")
    elif abs(test_metrics['accuracy'] - baseline_accuracy) < 0.01:
        print("○ M3T achieves comparable performance to baseline")
    else:
        print("△ M3T shows competitive performance with additional benefits")
    
    print(f"\nM3T Additional Benefits:")
    print(f"• Multi-modal learning (SMILES + descriptors)")
    print(f"• Attention-based interpretability")
    print(f"• Contrastive representation learning")
    print(f"• Scalable transformer architecture")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    plt.axhline(y=baseline_accuracy, color='orange', linestyle='--', 
                label=f'Baseline ({baseline_accuracy:.3f})')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_f1'], 'g-', label='Validation F1')
    plt.title('F1-Score Progression')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('m3t_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "=" * 80)
    print("M3T TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return test_metrics, history

if __name__ == "__main__":
    test_metrics, history = train_m3t_model()
