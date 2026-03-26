# Parameter Golf — Transformer Architecture & Training Guide

## ASCII Architecture Diagram

```
                         ┌─────────────────────────────┐
                         │      Input Token IDs         │
                         │     (batch, seq_len=1024)    │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │    Token Embedding (Tied)    │
                         │   nn.Embedding(1024, 512)    │
                         │  init: N(0, 0.005)           │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │     Post-Embed RMSNorm       │
                         └──────────────┬───────────────┘
                                        │
                                   x₀ = x  ◄──────────────────── (saved for residual mixing)
                                        │
              ┌─────────────────────────────────────────────────────────┐
              │                  ENCODER (Layers 0–3)                   │
              │                                                         │
              │  For each layer i in [0, 1, 2, 3]:                     │
              │                                                         │
              │    ┌───────────────────────────────────────┐            │
              │    │  Residual Mix: x = α·x + β·x₀        │            │
              │    │  (learned per-dim α,β weights)        │            │
              │    └───────────────────┬───────────────────┘            │
              │                        │                                │
              │                        ▼                                │
              │    ┌───────────────────────────────────────┐            │
              │    │           RMSNorm (attn)              │            │
              │    └───────────────────┬───────────────────┘            │
              │                        │                                │
              │                        ▼                                │
              │    ┌───────────────────────────────────────┐            │
              │    │    Causal Self-Attention (GQA)        │            │
              │    │                                       │            │
              │    │  Q: 512→512 (8 heads × 64d)          │            │
              │    │  K: 512→256 (4 KV heads × 64d)       │            │
              │    │  V: 512→256 (4 KV heads × 64d)       │            │
              │    │                                       │            │
              │    │  ┌─ QK RMSNorm ─┐                    │            │
              │    │  │ q = norm(q)   │                    │            │
              │    │  │ k = norm(k)   │                    │            │
              │    │  └──────────────-┘                    │            │
              │    │                                       │            │
              │    │  ┌── RoPE (base=10000) ──┐            │            │
              │    │  │ q = rotary(q)          │            │            │
              │    │  │ k = rotary(k)          │            │            │
              │    │  └───────────────────────-┘            │            │
              │    │                                       │            │
              │    │  q = q * q_gain  (init=1.5)          │            │
              │    │                                       │            │
              │    │  y = FlashAttention(q,k,v, causal)   │            │
              │    │  out = proj(y)  [512→512, zero_init] │            │
              │    └───────────────────┬───────────────────┘            │
              │                        │                                │
              │              x = x + attn_scale · out                  │
              │              (learned per-dim scale)                    │
              │                        │                                │
              │                        ▼                                │
              │    ┌───────────────────────────────────────┐            │
              │    │           RMSNorm (mlp)               │            │
              │    └───────────────────┬───────────────────┘            │
              │                        │                                │
              │                        ▼                                │
              │    ┌───────────────────────────────────────┐            │
              │    │          ReLU² MLP                    │            │
              │    │  fc:   512 → 1024 (2× expansion)     │            │
              │    │  act:  relu(x)²                      │            │
              │    │  proj: 1024 → 512  [zero_init]       │            │
              │    └───────────────────┬───────────────────┘            │
              │                        │                                │
              │              x = x + mlp_scale · out                   │
              │              (learned per-dim scale)                    │
              │                        │                                │
              │                ┌───────┴───────┐                       │
              │                │  save skip[i] │                       │
              │                └───────┬───────┘                       │
              │                        │                                │
              └────────────────────────┼────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────────────┐
              │                  DECODER (Layers 4–8)                   │
              │                                                         │
              │  For each layer j in [0, 1, 2, 3, 4]:                  │
              │                                                         │
              │    ┌───────────────────────────────────────┐            │
              │    │  Skip Connection (if available):      │            │
              │    │  x = x + skip_weight[j] · skip.pop() │            │
              │    │  (learned per-dim, connects enc↔dec)  │            │
              │    └───────────────────┬───────────────────┘            │
              │                        │                                │
              │          [ Same Block as Encoder ]                     │
              │          (ResidMix → Attn → MLP)                       │
              │                        │                                │
              └────────────────────────┼────────────────────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────────┐
                         │       Final RMSNorm          │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   LM Head (Tied Embedding)   │
                         │  logits = x @ E^T            │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │     Logit Soft-Capping       │
                         │  30 · tanh(logits / 30)      │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Cross-Entropy Loss         │
                         │   (next-token prediction)    │
                         └─────────────────────────────┘
```

---

## Architecture Design Choices Explained

### 1. Tiny Vocabulary (1024 tokens) with Tied Embeddings

```
Embedding: 1024 vocab × 512 dim = 524,288 parameters
```

The challenge imposes a **16 MB artifact size limit**. With a typical 32K+ vocabulary, the embedding table alone would consume most of the budget. A 1024-token SentencePiece BPE vocabulary keeps the embedding table at ~0.5M params. **Tying** the input embedding and output LM head (sharing the same weight matrix) halves this further in effective parameter cost. The embedding is initialized with a small std (0.005) to prevent dominating early training dynamics.

### 2. Grouped Query Attention (GQA): 8Q / 4KV

```
Q projections: 8 heads × 64 dim = 512 total
K projections: 4 heads × 64 dim = 256 total  (shared across 2 Q heads each)
V projections: 4 heads × 64 dim = 256 total
```

GQA halves the KV parameter count compared to standard multi-head attention. With a tight parameter budget, this frees capacity for more layers or wider MLPs. Each pair of query heads shares one KV head, which empirically retains most of the expressiveness of full MHA while being more parameter-efficient.

### 3. QK-Norm + Learnable Q-Gain

```python
q = rms_norm(q)      # stabilize attention logits
k = rms_norm(k)
q = q * q_gain       # learned per-head scalar, init=1.5
```

Normalizing Q and K before computing attention scores prevents logit explosion, especially important with small models where variance can be unstable. The learnable `q_gain` (initialized at 1.5) lets each head control its own attention sharpness — higher gain = sharper attention. This replaces the traditional `1/√d_k` scaling with a more flexible learned alternative.

### 4. RoPE (Rotary Position Embeddings)

```
Base frequency: 10,000
Applied to Q and K after normalization
```

RoPE encodes position via rotation matrices applied to Q/K pairs, enabling relative position awareness without adding parameters. It naturally supports length generalization and is more parameter-efficient than learned positional embeddings — critical when every parameter counts.

### 5. ReLU² Activation (Squared ReLU)

```python
def forward(self, x):
    x = torch.relu(self.fc(x))
    return self.proj(x.square())
```

Squared ReLU (`relu(x)²`) is computationally cheaper than GELU/SiLU (no exp/sigmoid) while providing strong sparsity. The squaring amplifies larger activations and kills small ones, creating a natural feature selection effect. This is borrowed from the modded-nanogpt tradition where it has been shown to work well at small scale.

### 6. 2× MLP Expansion (Instead of Standard 4×)

```
MLP hidden dim: 2 × 512 = 1024
```

The standard transformer uses 4× expansion. Halving this to 2× saves significant parameters, allowing those to be spent on more layers instead. At this model scale, width in the MLP is less important than depth for language modeling quality.

### 7. Encoder-Decoder Skip Connections (U-Net Style)

```
Layers 0-3: Encoder — saves activations as skip connections
Layers 4-8: Decoder — adds weighted skip connections from encoder

skip[0] connects Layer 3 ↔ Layer 4  (deepest)
skip[1] connects Layer 2 ↔ Layer 5
skip[2] connects Layer 1 ↔ Layer 6
skip[3] connects Layer 0 ↔ Layer 7
```

This is inspired by U-Net architectures. The first half of the network (encoder) stores intermediate representations, and the second half (decoder) incorporates them via learned `skip_weights`. This creates shortcut paths for gradient flow and allows the decoder to access multi-resolution features from the encoder — effectively getting more depth of processing without the vanishing gradient cost.

### 8. Residual Mixing (x₀ Shortcut)

```python
x = α · x + β · x₀    # learned per-dimension α, β
```

Every block mixes the current hidden state `x` with the original post-embedding representation `x₀`. This is a form of "shortcut to input" that prevents the model from forgetting low-level token features as it processes through deep layers. The mixing weights are learned per dimension, allowing the model to decide which features should persist from the input.

### 9. Per-Dimension Residual Scaling

```python
x = x + attn_scale · attn_output     # learned per-dim
x = x + mlp_scale  · mlp_output      # learned per-dim
```

Instead of a fixed residual connection, each sub-layer's contribution is scaled by a learned per-dimension vector (initialized to 1.0). Combined with zero-initialization of output projections, this means each block starts as an identity function and gradually learns to contribute, stabilizing early training.

### 10. Zero-Init Output Projections

```
Attention proj (512→512): initialized to zeros
MLP proj (1024→512):      initialized to zeros
```

The output projections of both attention and MLP are initialized to zero. At initialization, each block is effectively a no-op (identity via residual). Training then gradually "turns on" each layer, which is crucial for stable training of deep networks without warmup tricks.

### 11. Logit Soft-Capping

```python
logits = 30.0 * tanh(logits / 30.0)
```

Caps logits to the range [-30, 30] using a smooth tanh function (from Gemma 2). This prevents extreme confidence in predictions, acts as implicit regularization, and stabilizes training — especially important with the small vocabulary where the model might otherwise become overconfident on common tokens.

### 12. CastedLinear (FP32 Weights, BF16 Compute)

```python
class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), ...)
```

Weights are stored in FP32 for optimizer precision but cast to BF16 at compute time. This gives the optimizer full precision for small updates while keeping forward/backward passes fast with BF16 matmuls.

---

## Training Infrastructure

### Optimizer: Muon + Adam Split

The training uses a **three-way optimizer split**, assigning different optimizers to different parameter types:

| Parameter Type | Optimizer | Learning Rate | Rationale |
|---|---|---|---|
| Token embeddings (tied) | Adam | 0.05 | Embeddings need stable, adaptive updates |
| 2D weight matrices | **Muon** | 0.04 | Orthogonal updates are better for matrix-shaped params |
| 1D params & scalars | Adam | 0.04 | Standard adaptive optimization for small params |
| LM head (if untied) | Adam | 0.008 | Lower LR for output projection stability |

**Muon** is a novel optimizer that orthogonalizes gradient matrices using a fast Newton-Schulz iteration (5 steps). Instead of following the raw gradient, it finds the nearest orthogonal matrix to the gradient update. This implicitly regularizes weight norms and has been shown to outperform Adam for matrix-shaped parameters in small language models.

### Learning Rate Schedule

```
Step 0          Step 20                    Step 18,800     Step 20,000
  |── warmup ──|────── full LR ──────────|── warmdown ──|
  |  (20 steps) |                          | (1200 steps) |
```

- **Warmup** (20 steps): Primes CUDA compiled kernels; model state is reset after warmup
- **Constant phase**: Full learning rate for the bulk of training
- **Linear warmdown**: LR decreases linearly to 0 over the last 1200 steps
- **Wallclock cap**: 10-minute hard limit on 8×H100; warmdown can also be triggered by remaining wall time

### Muon Momentum Warmup

```
Step 0                 Step 500
  |── momentum ramp ──|── constant ──
  0.85 ──────────────→ 0.95
```

Muon's momentum starts at 0.85 and linearly ramps to 0.95 over 500 steps. Lower initial momentum allows the optimizer to explore more freely early in training before settling into a faster-converging high-momentum regime.

### Data Pipeline

```
FineWeb 10B dataset (pre-tokenized, 1024-token SentencePiece BPE)
        │
        ▼
  80 binary shards (fineweb_train_*.bin)
        │
        ▼
  TokenStream: sequential read, wrap-around
        │
        ▼
  DistributedTokenLoader: disjoint spans per GPU rank
        │
        ▼
  Batch: 524,288 tokens → reshape to (512, 1024) sequences
        │
  Target: shift by 1 token (standard causal LM)
```

- **Batch size**: 524,288 tokens per gradient step (~512 sequences of length 1024)
- **Gradient accumulation**: 8 micro-steps (divided across GPUs)
- **No shuffling**: Deterministic sequential streaming through shards
- **Shard format**: Custom binary with 1024-byte header + uint16 tokens

### Mixed Precision & Compilation

- **Forward/backward**: BF16 via `torch.autocast`
- **Weights**: FP32 (cast to BF16 at matmul time via `CastedLinear`)
- **Control params**: Always FP32 (scales, gains, mixing weights)
- **torch.compile**: Full-graph, static-shape compilation for maximum kernel fusion
- **Flash Attention**: Enabled (cuDNN SDP and math SDP disabled)
- **TF32**: Enabled for matmuls and cuDNN ops

### Post-Training Quantization

```
Trained Model (FP32/BF16)
        │
        ▼
  INT8 Quantization
  ├── 2D matrices: per-row INT8 + FP16 scales
  ├── 1D vectors:  per-tensor INT8 + FP32 scale
  └── Small tensors (≤64K elements): stored as FP16 directly
        │
        ▼
  ZLIB Compression (level 9)
        │
        ▼
  final_model.int8.ptz  (must be ≤ 16 MB)
```

The 16 MB limit is for the **compressed quantized model**. INT8 per-row quantization with 99.99984 percentile clipping preserves quality with minimal loss. A roundtrip validation confirms the quantized model matches the original.

### Test-Time Training (LoRA)

At evaluation, the model adapts to each validation document via per-document LoRA adapters:

```
For each document in validation set:
  1. Initialize fresh LoRA adapters (rank=8):
     - Q projection: rank-8 LoRA per layer
     - V projection: rank-8 LoRA per layer
     - LM head: rank-8 LoRA
  2. Split document into chunks (256 tokens)
  3. For each chunk:
     - Forward pass with LoRA applied
     - Backprop through LoRA params only (base model frozen)
     - Accumulate BPB metrics
     - Adam update on LoRA params (lr=0.01)
```

This is a form of **test-time computation** — the model gets to "study" each document before being evaluated on it. The LoRA rank of 8 keeps the adaptation lightweight while allowing meaningful per-document specialization.

### Evaluation Metric: Bits-Per-Byte (BPB)

```
BPB = (val_loss / ln(2)) × (total_tokens / total_bytes)
```

BPB is **tokenizer-agnostic** — it measures compression efficiency in bits per UTF-8 byte regardless of vocabulary size. This is critical because the challenge allows custom tokenizers; BPB ensures fair comparison. The byte count accounts for SentencePiece's leading-space markers (`▁`) and control tokens.

---

## Model Summary

| Component | Value |
|---|---|
| Vocabulary | 1,024 tokens (SentencePiece BPE) |
| Model dimension | 512 |
| Number of layers | 9 (4 encoder + 5 decoder) |
| Attention heads | 8 query, 4 KV (GQA) |
| Head dimension | 64 |
| MLP hidden dim | 1,024 (2× expansion) |
| Activation | ReLU² |
| Positional encoding | RoPE (base 10,000) |
| Normalization | RMSNorm (parameterless) |
| Embeddings | Tied (input = output) |
| Logit capping | 30.0 (tanh soft-cap) |
| Sequence length | 1,024 |
| Training tokens/step | 524,288 |
| Max iterations | 20,000 |
| Training time limit | 10 min on 8×H100 |
| Artifact size limit | 16 MB (compressed INT8) |
| SOTA BPB | 1.1748 |
