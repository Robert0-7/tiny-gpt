# Tiny GPT from scratch

A character-level GPT (decoder-only transformer) built and trained from scratch in PyTorch, step by step in a Jupyter notebook. Trained on Tiny Shakespeare.

Based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his "Let's build GPT" lecture.

## Model
- 10.77M parameters: 6 layers, 6 attention heads, 384-dim embeddings
- Context window: 256 characters, vocabulary: 65 characters
- Causal self-attention, pre-LayerNorm, residual connections, weight tying, GPT-2-style init (std 0.02)
- Mixed-precision (fp16) training with AdamW and gradient clipping

## Results
Trained on an **RTX 3050 Laptop GPU (4GB)** in about 8.6 minutes (5000 steps).

| Step | Train loss | Val loss |
|------|-----------|----------|
| 0    | 4.244     | 4.244    |
| 1000 | 1.535     | 1.716    |
| 2500 | 1.242     | 1.514    |
| 4000 | 1.105     | **1.469** |
| 5000 | 1.027     | 1.485    |

Val loss bottoms out around step 4000 while train loss keeps falling, which is overfitting on the small (~1MB) dataset.

Sample output (prompt: `ROMEO:`):
```
ROMEO:
But so twenty that I have sent not to do,
To content them, the room of her complaints,
And give me peace with a brancher straight
To sweet forth thy subjects' tedious words:
```

## What I learned
- Without proper weight initialisation, the starting loss was 249 instead of ln(65) ≈ 4.17
- How temperature and top-k sampling shape generated text
- Reading train/val curves to spot overfitting

## Run it
```
pip install torch numpy jupyter
```
Open `tiny_gpt.ipynb` and run the cells top to bottom.