# Demo 2: Understanding Attention with nanoGPT

## Setup and Imports

```python
# Install required packages
%pip install -q torch matplotlib seaborn python-dotenv psutil

# Import libraries
import os
import torch
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch.nn.functional as F

# Set up paths
notebook_dir = os.getcwd()
nanoGPT_dir = os.path.join(notebook_dir, 'nanoGPT')

# Add nanoGPT to Python path
import sys
sys.path.append(nanoGPT_dir)

# Import model
from nanoGPT.model import GPTConfig, GPT
```

## Model Configuration

```python
# Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Training hyperparameters
config = {
    'n_layer': 4,  # Reduced layers
    'n_head': 4,   # Reduced heads
    'n_embd': 256, # Reduced embedding
    'dropout': 0.0,
    'bias': False,
    'max_iters': 300,
    'batch_size': 16,
    'gradient_accumulation_steps': 15,
    'block_size': 256,  # Smaller block size
    'learning_rate': 6e-4,
    'weight_decay': 1e-1,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,
}

# Initialize model
model_config = GPTConfig(
    block_size=config['block_size'],
    vocab_size=65,  # ASCII characters
    n_layer=config['n_layer'],
    n_head=config['n_head'],
    n_embd=config['n_embd'],
    dropout=config['dropout'],
    bias=config['bias']
)

# Check if we have a saved model
checkpoint_path = os.path.join(nanoGPT_dir, 'out-shakespeare-char', 'ckpt.pt')
if os.path.exists(checkpoint_path):
    print("Loading saved model...")
    checkpoint = torch.load(checkpoint_path)
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()  # Set to evaluation mode
else:
    print("Training new model...")
    model = GPT(model_config)
    model.to(device)
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config['weight_decay'],
        learning_rate=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
        device_type=device
    )
    
    # Training loop
    print("Starting training...")
    model.train()
    last_time = time.time()
    
    # Pre-load data
    data_dir = os.path.join(nanoGPT_dir, 'data', 'shakespeare_char')
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    
    for iter_num in range(config['max_iters']):
        # Get batch
        ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
        x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
        
        # Move to device
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        
        # Logging
        if iter_num % 10 == 0:
            current_time = time.time()
            iter_time = current_time - last_time
            last_time = current_time
            progress = (iter_num + 1) / config['max_iters'] * 100
            if device == 'mps':
                gpu_percent = psutil.cpu_percent(interval=0.1)
                print(f"iter {iter_num}: loss {loss.item():.4f} | time {iter_time:.2f}s | {progress:.1f}% | mpu {gpu_percent:.1f}%")
            else:
                print(f"iter {iter_num}: loss {loss.item():.4f} | time {iter_time:.2f}s | {progress:.1f}%")
    
    # Save the trained model
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_config.__dict__,
        'iter_num': iter_num,
        'best_val_loss': loss.item(),
    }
    torch.save(checkpoint, checkpoint_path)
```

## Save Model

```python
# Save the trained model
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_config.__dict__,
    'iter_num': iter_num,
    'best_val_loss': loss.item(),
}
torch.save(checkpoint, os.path.join(nanoGPT_dir, 'out-shakespeare-char', 'ckpt.pt'))
```

## Load Saved Model

```python
# Load the saved model
checkpoint = torch.load(os.path.join(nanoGPT_dir, 'out-shakespeare-char', 'ckpt.pt'))
model_config = GPTConfig(**checkpoint['model_args'])
model = GPT(model_config)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()  # Set to evaluation mode for visualization
```

## Attention Visualization Setup

```python
def get_attention_patterns(model, x):
    """Extract attention patterns from the model without modifying its structure."""
    B, T = x.shape
    assert T <= model.config.block_size, f"Cannot forward sequence of length {T}, block size is only {model.config.block_size}"
    
    # forward the GPT model
    tok_emb = model.transformer.wte(x) # token embeddings of shape (b, t, n_embd)
    pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0) # shape (1, t)
    pos_emb = model.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
    x = model.transformer.drop(tok_emb + pos_emb)
    
    # Store attention weights
    attention_weights = []
    
    # Forward through each block and capture attention
    for block in model.transformer.h:
        # Get query, key, value projections
        qkv = block.attn.c_attn(block.ln_1(x))
        q, k, v = qkv.split(model.config.n_embd, dim=2)
        
        # Reshape for attention
        B, T, C = x.shape
        k = k.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(1, 2)
        q = q.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(1, 2)
        v = v.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(1, 2)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask if not using Flash Attention
        if not block.attn.flash:
            att = att.masked_fill(block.attn.bias[:,:,:T,:T] == 0, float('-inf'))
        else:
            # Create causal mask for Flash Attention
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        
        # Store attention weights
        attention_weights.append(att)
        
        # Continue with normal forward pass
        x = x + block.attn(block.ln_1(x))
        x = x + block.mlp(block.ln_2(x))
    
    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    
    return logits, attention_weights

def visualize_attention(text, model, layer_idx=0, head_idx=0):
    """Visualize attention patterns for a given input text."""
    # Tokenize input
    chars = list(text)
    x = torch.tensor([ord(c) for c in chars], dtype=torch.long).unsqueeze(0)
    
    # Move to device
    x = x.to(model.transformer.wte.weight.device)
    
    # Get attention weights
    with torch.no_grad():
        logits, attention_weights = get_attention_patterns(model, x)
    
    # Get attention for specified layer and head
    attn = attention_weights[layer_idx][0, head_idx]
    
    # Create attention heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attn.cpu().numpy(), cmap='viridis')
    plt.xticks(range(len(chars)), chars, rotation=90)
    plt.yticks(range(len(chars)), chars)
    plt.title(f'Attention Pattern (Layer {layer_idx}, Head {head_idx})')
    plt.colorbar()
    plt.show()

# Example text from Shakespeare
text = "To be, or not to be, that is the question"
visualize_attention(text, model)
```

## Multi-Head Analysis

```python
def analyze_attention_heads(text, model, layer_idx=0):
    """Analyze attention patterns across all heads in a layer."""
    # Tokenize input
    chars = list(text)
    x = torch.tensor([ord(c) for c in chars], dtype=torch.long).unsqueeze(0)
    
    # Move to device
    x = x.to(model.transformer.wte.weight.device)
    
    # Get attention weights
    with torch.no_grad():
        logits, attention_weights = model.get_attention_weights(x)
    
    # Create subplot for each head
    n_heads = model.config.n_head
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'Attention Patterns Across Heads (Layer {layer_idx})')
    
    for i in range(n_heads):
        row = i // 2
        col = i % 2
        attn = attention_weights[layer_idx][0, i]
        
        im = axes[row, col].imshow(attn.cpu().numpy(), cmap='viridis')
        axes[row, col].set_xticks(range(len(chars)))
        axes[row, col].set_xticklabels(chars, rotation=90)
        axes[row, col].set_yticks(range(len(chars)))
        axes[row, col].set_yticklabels(chars)
        axes[row, col].set_title(f'Head {i}')
        plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    plt.show()

# Analyze attention heads
text = "All the world's a stage, and all the men and women merely players"
analyze_attention_heads(text, model)