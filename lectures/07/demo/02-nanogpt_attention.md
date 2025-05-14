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

# Karpathy's miniature Shakespeare config (baby GPT, fast for laptops/MacBooks)
config = {
    'n_layer': 6,      # 6 layers
    'n_head': 6,       # 6 attention heads
    'n_embd': 384,     # 384 embedding size
    'dropout': 0.2,    # regularization
    'bias': False,
    'max_iters': 500,  # training iterations
    'batch_size': 64,  # larger batch size for small model
    'gradient_accumulation_steps': 1,
    'block_size': 256, # context of up to 256 previous characters
    'learning_rate': 1e-3, # higher learning rate for small model
    'weight_decay': 1e-1,
    'beta1': 0.9,
    'beta2': 0.99,     # higher beta2 for small batches
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
checkpoint_path = os.path.join(nanoGPT_dir, 'out-shakespeare-char', 'model.pt')
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
torch.save(checkpoint, os.path.join(nanoGPT_dir, 'out-shakespeare-char', 'model.pt'))
```

## Load Saved Model

```python
# Load the saved model
checkpoint = torch.load(os.path.join(nanoGPT_dir, 'out-shakespeare-char', 'model.pt'))
model_config = GPTConfig(**checkpoint['model_args'])
model = GPT(model_config)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()  # Set to evaluation mode for visualization
```

## Attention Visualization Setup

Short or simple sentences often produce diagonal attention patterns, showing that each token mostly attends to itself or its immediate neighbors. By visualizing only a subset of the attention matrix for a short input, we can make the plot readable and see how the model distributes attention across tokens. This is especially useful for character-level models, where long inputs make the plot crowded.

```python
def get_attention_patterns(model, x):
    """Extract attention patterns from the model without modifying its structure."""
    B, T = x.shape
    assert T <= model.config.block_size, f"Cannot forward sequence of length {T}, block size is only {model.config.block_size}"
    # Forward the GPT model
    tok_emb = model.transformer.wte(x) # token embeddings of shape (b, t, n_embd)
    pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0) # shape (1, t)
    pos_emb = model.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
    x = model.transformer.drop(tok_emb + pos_emb)
    attention_weights = []
    # Forward through each block and capture attention
    for block in model.transformer.h:
        qkv = block.attn.c_attn(block.ln_1(x))
        q, k, v = qkv.split(model.config.n_embd, dim=2)
        B, T, C = x.shape
        k = k.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(1, 2)
        q = q.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(1, 2)
        v = v.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if not block.attn.flash:
            att = att.masked_fill(block.attn.bias[:,:,:T,:T] == 0, float('-inf'))
        else:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        attention_weights.append(att)
        x = x + block.attn(block.ln_1(x))
        x = x + block.mlp(block.ln_2(x))
    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits, attention_weights

def visualize_attention_subset(text, model, layer_idx=0, head_idx=0, max_tokens=20):
    """Visualize a subset of the attention pattern for readability."""
    # Use only the first max_tokens characters
    chars = list(text)[:max_tokens]
    x = torch.tensor([ord(c) for c in chars], dtype=torch.long).unsqueeze(0)
    x = x.to(model.transformer.wte.weight.device)
    with torch.no_grad():
        logits, attention_weights = get_attention_patterns(model, x)
    attn = attention_weights[layer_idx][0, head_idx]
    plt.figure(figsize=(6, 5))
    plt.imshow(attn.cpu().numpy(), cmap='viridis')
    plt.xticks(range(len(chars)), chars, rotation=90)
    plt.yticks(range(len(chars)), chars)
    plt.title(f'Attention Pattern (Layer {layer_idx}, Head {head_idx}) [First {max_tokens} chars]')
    plt.colorbar()
    plt.show()
```

### Short Input Example

This plot shows attention for a short phrase. Look for diagonal patterns, which indicate local attention (each character mostly attends to itself or its neighbors).

```python
short_text = "To be, or not to be"
visualize_attention_subset(short_text, model)
```

### Complex Input Example

With a longer, more complex input, attention patterns can reveal longer-range dependencies, repeated word focus, or punctuation effects. The plot may be more crowded, but you may spot heads that focus on repeated words or punctuation.

```python
complex_text = (
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles "
    "And by opposing end them."
)
visualize_attention_subset(complex_text, model, max_tokens=40)
```

## Multi-Head Analysis (Subset)

Visualizing all attention heads for a short input helps us compare how different heads focus on different parts of the sequence. Some heads may focus on local context, while others may capture longer-range dependencies or special characters.

```python
def analyze_attention_heads_subset(text, model, layer_idx=0, max_tokens=20):
    """Analyze attention patterns across all heads in a layer for a subset of tokens."""
    chars = list(text)[:max_tokens]
    x = torch.tensor([ord(c) for c in chars], dtype=torch.long).unsqueeze(0)
    x = x.to(model.transformer.wte.weight.device)
    with torch.no_grad():
        logits, attention_weights = get_attention_patterns(model, x)
    n_heads = model.config.n_head
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    fig.suptitle(f'Attention Patterns Across Heads (Layer {layer_idx}) [First {max_tokens} chars]')
    axes = axes.flatten() if n_heads > 1 else [axes]
    for i in range(n_heads):
        ax = axes[i]
        attn = attention_weights[layer_idx][0, i]
        im = ax.imshow(attn.cpu().numpy(), cmap='viridis')
        ax.set_xticks(range(len(chars)))
        ax.set_xticklabels(chars, rotation=90)
        ax.set_yticks(range(len(chars)))
        ax.set_yticklabels(chars)
        ax.set_title(f'Head {i}')
        plt.colorbar(im, ax=ax)
    for j in range(n_heads, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
```

### Multi-Head, Short Input

Compare how different heads focus on the short phrase. Most heads will show local/diagonal attention, but some may focus on punctuation or repeated characters.

```python
analyze_attention_heads_subset(short_text, model)
```

### Multi-Head, Complex Input

With a longer input, some heads may focus on repeated words, punctuation, or long-range dependencies. The plot is more crowded, but you may spot heads that specialize in certain patterns.

```python
analyze_attention_heads_subset(complex_text, model, max_tokens=40)
```

## Multi-Head Analysis

```python
def analyze_attention_heads(text, model, layer_idx=0):
    """Analyze attention patterns across all heads in a layer."""
    # Tokenize input
    chars = list(text)
    x = torch.tensor([ord(c) for c in chars], dtype=torch.long).unsqueeze(0)
    x = x.to(model.transformer.wte.weight.device)
    
    # Get attention weights using the custom function
    with torch.no_grad():
        logits, attention_weights = get_attention_patterns(model, x)
    
    n_heads = model.config.n_head
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    fig.suptitle(f'Attention Patterns Across Heads (Layer {layer_idx})')
    
    axes = axes.flatten() if n_heads > 1 else [axes]
    for i in range(n_heads):
        ax = axes[i]
        attn = attention_weights[layer_idx][0, i]
        im = ax.imshow(attn.cpu().numpy(), cmap='viridis')
        ax.set_xticks(range(len(chars)))
        ax.set_xticklabels(chars, rotation=90)
        ax.set_yticks(range(len(chars)))
        ax.set_yticklabels(chars)
        ax.set_title(f'Head {i}')
        plt.colorbar(im, ax=ax)
    # Hide any unused subplots
    for j in range(n_heads, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

# Use the same complex text for multi-head analysis
analyze_attention_heads(text, model)