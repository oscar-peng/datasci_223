# Demo 2: Understanding Attention with nanoGPT

This demo explores the attention mechanism in transformers using Andrej Karpathy's nanoGPT implementation. We'll train a small GPT model on Shakespeare text and analyze how attention heads learn patterns in the text.

## Setup

First, let's clone and set up nanoGPT:

```bash
# Clone nanoGPT repository
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Let's prepare the Shakespeare dataset:

```bash
# Download and prepare Shakespeare dataset
python data/shakespeare_char/prepare.py

# Display sample of the prepared data
head -n 5 data/shakespeare_char/input.txt
```

## Training Configuration

We'll use a small model configuration to demonstrate the concepts:

```bash
# Create a custom config file
cat > config/shakespeare_char_custom.py << 'EOL'
out_dir = 'out-shakespeare-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2
bias = False

# optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4
EOL
```

## Training the Model

Let's train our small GPT model:

```bash
# Train the model
python train.py config/shakespeare_char_custom.py
```

Now let's load and analyze the model:

```python
# Load the trained model
import torch
from model import GPTConfig, GPT

# Initialize model
config = GPTConfig(
    block_size=256,
    vocab_size=65,  # ASCII characters
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.2,
    bias=False
)
model = GPT(config)
model.load_state_dict(torch.load('out-shakespeare-char/model.pt'))
model.eval()
```

## Understanding Attention

Let's analyze how the attention heads work:

```python
def visualize_attention(text, model, layer_idx=0, head_idx=0):
    """
    Visualize attention patterns for a given input text.
    """
    # Tokenize input
    chars = list(text)
    x = torch.tensor([ord(c) for c in chars], dtype=torch.long).unsqueeze(0)
    
    # Get attention weights
    with torch.no_grad():
        logits, attention_weights = model(x, return_attention=True)
    
    # Get attention for specified layer and head
    attn = attention_weights[layer_idx][0, head_idx]
    
    # Create attention heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attn.numpy(), cmap='viridis')
    plt.xticks(range(len(chars)), chars, rotation=90)
    plt.yticks(range(len(chars)), chars)
    plt.title(f'Attention Pattern (Layer {layer_idx}, Head {head_idx})')
    plt.colorbar()
    plt.show()

# Example text from Shakespeare
text = "To be, or not to be, that is the question"
visualize_attention(text, model)
```

## Analyzing Different Attention Heads

Let's look at how different heads specialize in different patterns:

```python
def analyze_attention_heads(text, model, layer_idx=0):
    """
    Analyze attention patterns across all heads in a layer.
    """
    # Tokenize input
    chars = list(text)
    x = torch.tensor([ord(c) for c in chars], dtype=torch.long).unsqueeze(0)
    
    # Get attention weights
    with torch.no_grad():
        logits, attention_weights = model(x, return_attention=True)
    
    # Create subplot for each head
    n_heads = model.config.n_head
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'Attention Patterns Across Heads (Layer {layer_idx})')
    
    for i in range(n_heads):
        row = i // 2
        col = i % 2
        attn = attention_weights[layer_idx][0, i]
        
        im = axes[row, col].imshow(attn.numpy(), cmap='viridis')
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
```

## Text Generation

Let's see how our model generates text:

```python
def generate_text(model, prompt, max_tokens=100, temperature=0.8):
    """
    Generate text using the trained model.
    """
    # Tokenize prompt
    chars = list(prompt)
    x = torch.tensor([ord(c) for c in chars], dtype=torch.long).unsqueeze(0)
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get predictions
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_char = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_char], dim=1)
            
            # Stop if we generate a newline
            if next_char.item() == ord('\n'):
                break
    
    # Convert back to text
    generated = ''.join([chr(i) for i in x[0].tolist()])
    return generated

# Generate some text
prompt = "ROMEO: "
generated = generate_text(model, prompt)
print(generated)
```

## Key Takeaways

1. **Attention Mechanism**
   - Allows model to focus on different parts of input
   - Each head can learn different patterns
   - Enables parallel processing of sequences

2. **Model Architecture**
   - Multiple layers of attention
   - Each layer has multiple attention heads
   - Residual connections and layer normalization

3. **Training Process**
   - Character-level prediction
   - Autoregressive generation
   - Temperature controls randomness

4. **Healthcare Applications**
   - Can be adapted for medical text
   - Useful for clinical note analysis
   - Potential for medical report generation 