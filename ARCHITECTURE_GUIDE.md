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

---

# Next-Generation Architecture: Research & Compound Improvement Strategy

## Executive Summary

Eight parallel research tracks were investigated to push beyond the current SOTA (actual best: **1.1428 BPB** from `10L_Int5MLP_MuonWD04_SWA50`). The key insight: the constraint is **compressed artifact size**, not raw parameter count. Techniques that improve compression ratio effectively give "free" parameters.

**Target: ~1.125 BPB** (pessimistic: 1.138, optimistic: 1.115)

---

## Technique Rankings (by Impact/Effort)

| Rank | Technique | Est. BPB Gain | Compressed Size | Flash Attn? | Priority |
|------|-----------|--------------|-----------------|-------------|----------|
| 1 | **2:4 Sparse 4x MLP + zstd** | -0.023 to -0.033 | Saves 2.3MB | Yes | HIGHEST |
| 2 | **MLA (Latent KV Compression)** | -0.012 to -0.025 | Saves 1.79M params | Yes | HIGH |
| 3 | **Value Residual Learning** | -0.010 to -0.020 | +36 params | Yes | HIGH |
| 4 | **Low-Rank MoE** | -0.010 to -0.025 | Saves ~4.8MB | Yes | HIGH |
| 5 | **QAT INT4/5 MLP** | -0.005 | Saves ~1MB | Yes | MEDIUM |
| 6 | **Differential Attention** | -0.000 to -0.005 | +72 params | NO (loses Flash) | LOW |
| 7 | **Engram Memory Tokens** | -0.001 to -0.005 | +17K params | Risky | LOW |

---

## 1. QAT + Model Scaling (INT4/5/6 Fake Quantization)

### Core Idea
Train with fake quantization (Straight-Through Estimator) so the model learns to be robust to low-bit quantization. This allows packing MORE parameters into 16MB.

### Key Findings
- Current SOTA already uses INT5 MLP / INT6 attention with QAT
- **INT4 MLP is feasible** thanks to ReLU^2 sparsity (~50% zeros mask quantization noise)
- Progressive warmup: INT6 (0-30%) -> INT5 (30-60%) -> INT4 (60-100%)
- Muon produces well-conditioned weight distributions that quantize better
- INT4 saves ~1MB -> room for an extra layer

### Implementation (STE Fake Quantization)
```python
class CastedLinear(nn.Linear):
    _qat: bool = False
    _qat_clip: int = 127  # INT8 default

    def forward(self, x):
        w = self.weight
        if self._qat and self.training and w.ndim == 2:
            w_f = w.float()
            amax = w_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = amax / self._qat_clip
            q = (w_f / scale).round().clamp(-(self._qat_clip + 1), self._qat_clip)
            w_q = q * scale
            w = w + (w_q - w_f).detach()  # STE: forward uses quantized, backward flows through original
        return F.linear(x, w.to(x.dtype), self.bias)

# Configuration per layer type:
# MLP:  _qat_clip = 15  (INT5: [-16, 15])
# Attn: _qat_clip = 31  (INT6: [-32, 31])
```

### Parameter Budget at INT4 MLP / INT6 Attention
```
Per-layer attn (INT6, 0.75 B/w):  786,432 x 0.75 =  589,824 bytes
Per-layer MLP 3x (INT4, 0.5 B/w): 1,572,864 x 0.5 =  786,432 bytes
Per-layer total:                                     1,376,256 bytes
10 layers:                                          13,762,560 bytes
+ FP16 embed + BigramHash + metadata:               ~2,500,000 bytes
Total:                                             ~16,260,000 bytes  (fits!)
```

### Estimated Improvement: -0.005 BPB over SOTA

---

## 2. Structured Sparsity (2:4) + Advanced Compression

### Core Idea
In each group of 4 consecutive MLP weights, zero out the 2 smallest. A 4x expansion MLP with 50% sparsity = same FLOPs as dense 2x but far more expressive. The zeros compress nearly for free.

### Key Findings
- **4x MLP with 2:4 sparsity**: same compute as dense 2x, but 2x the hidden dimension
- Bitmask (1 bit/weight) compresses extremely well with zstd (regular patterns)
- Combined with INT6 on nonzero values: ~3.6 effective bits/weight
- **Saves ~2.3MB** vs SOTA -> funds 2 extra layers (12L total)
- Delta coding between adjacent layers saves additional ~5%

### Implementation
```python
def compute_2_4_mask(weight):
    """Keep the 2 largest magnitude weights per group of 4."""
    w = weight.detach().abs()
    w4 = w.reshape(w.shape[0], -1, 4)
    _, topk = w4.topk(2, dim=-1)
    mask = torch.zeros_like(w4)
    mask.scatter_(-1, topk, 1.0)
    return mask.reshape_as(w)

# Training: apply mask via STE during warmdown
# Export: store bitmask + nonzero INT6 values + zstd-22
```

### Compression Budget
```
Dense 3x MLP INT5 + zstd (SOTA):           ~8,200KB for 10L
Sparse 4x MLP INT6 + bitmask + zstd:       ~7,200KB for 10L (saves 1MB)
Sparse 4x MLP INT6 + bitmask + delta + zstd: ~6,400KB for 10L (saves 1.8MB)
```

### Estimated Improvement: -0.023 to -0.033 BPB (with extra layers from savings)

---

## 3. Multi-Latent Attention (MLA)

### Core Idea
Replace GQA's separate K,V projections with a shared low-rank KV compression bottleneck. Inspired by DeepSeek-V2/V3.

### Architecture
```
Standard GQA:                    MLA (Decoupled RoPE):
x -> W_k [512,256] -> K          x -> W_kv_down [512,128] -> latent
x -> W_v [512,256] -> V               latent -> W_k_up [128,192] -> K_content
                                       latent -> W_v_up [128,256] -> V
                                  x -> W_k_rope [512,64]  -> K_position
                                  Q = cat(Q_content, Q_rope)  [48d + 16d = 64d]
                                  K = cat(K_content, K_rope)  [48d + 16d = 64d]
```

### Key Findings
- **Saves 198,656 params/layer** (16.7% of attention params)
- 9 layers x 198K = **1.79M params freed** -> enough for a 10th or 11th layer
- **Fully Flash Attention compatible** (head_dim stays 64)
- All MLA-specific matrices are < 65K params -> stored as FP16 passthrough -> **zero quantization noise**
- Content/RoPE split preserves positional encoding quality
- LoRA TTT compatible with minimal changes

### Parameter Savings Per Layer
```
GQA:  W_q[512,512] + W_k[512,256] + W_v[512,256] + W_o[512,512] = 786,432
MLA:  W_q[512,512] + W_kv_dn[512,128] + W_k_up[128,192]
      + W_v_up[128,256] + W_k_rope[512,64] + W_o[512,512]   = 587,776
                                                       Saved:   198,656/layer
```

### Estimated Improvement: -0.012 to -0.025 BPB (including freed params for depth)

---

## 4. Differential Attention

### Core Idea
Split each attention head into two sub-heads, compute two attention patterns, subtract one from the other to cancel noise: `attn = softmax(Q1@K1^T) - lambda * softmax(Q2@K2^T)`

### Key Findings
- **Zero extra parameters** (just 8 lambda scalars per layer)
- Mathematically: `(A1 - lambda*A2) @ V = A1@V - lambda*A2@V` (distributive property)
- BUT: SDPA requires matching head_dim for Q/K/V — splitting creates dimension mismatch
- **Loses Flash Attention** unless a custom Triton kernel is written
- At seq_len=1024, noise cancellation benefit is limited (scales with longer sequences)
- ~10% throughput loss may offset quality gains

### Verdict
**Medium-low priority for this competition.** The throughput penalty from losing Flash Attention likely offsets the small quality gain at seq_len=1024. Would be compelling with a custom Triton kernel or longer sequences.

### If Implementing (Manual Attention at seq=1024 is feasible)
```python
# After QK-norm and RoPE on full 64-dim:
q1, q2 = q[..., :32], q[..., 32:]
k1, k2 = k[..., :32], k[..., 32:]
k1 = k1.repeat_interleave(gqa_reps, dim=1)  # expand for GQA
k2 = k2.repeat_interleave(gqa_reps, dim=1)
v_exp = v.repeat_interleave(gqa_reps, dim=1)

s1 = (q1 @ k1.transpose(-2,-1)) * (32 ** -0.5)
s2 = (q2 @ k2.transpose(-2,-1)) * (32 ** -0.5)
s1 = s1 + causal_mask; s2 = s2 + causal_mask

A1 = F.softmax(s1, dim=-1)
A2 = F.softmax(s2, dim=-1)
lam = torch.sigmoid(self.lambda_raw)[None, :, None, None]
y = (A1 - lam * A2) @ v_exp  # (bsz, heads, seq, 64)
```

---

## 5. Value Residual Learning

### Core Idea
Store the first layer's Value representation and blend it into all subsequent layers, preventing "value rank collapse" in deep transformers.

### Key Findings
- **Only 36 new parameters** (per-head learned alpha: 9 layers x 4 KV heads)
- V_0 captured from layer 0, blended via `V_i = (1-sigmoid(alpha_i)) * V_current + sigmoid(alpha_i) * V_0`
- **Complementary** with existing U-Net skips (skips work on hidden state, VR works inside attention)
- **Critical for deeper models** — value collapse worsens with depth
- Initialized with small alpha (sigmoid(-3) ~ 0.05) so layers start near normal
- 4 attention sink tokens (2,048 params) absorb "junk" attention — works naturally with causal masking

### Implementation
```python
# In GPT.__init__:
self.value_resid_alpha = nn.Parameter(
    torch.linspace(-3.0, -1.0, num_layers * num_kv_heads).reshape(num_layers, num_kv_heads)
)

# In attention forward: (layer i > 0)
alpha = torch.sigmoid(self.value_resid_alpha[i])[None, :, None, None]
v = (1.0 - alpha) * v + alpha * V_0  # V_0 from layer 0
```

### Why This Compounds with Everything
- VR operates **inside attention** (orthogonal to U-Net skips on hidden state)
- Prevents collapse that worsens with depth -> enables benefit of extra layers from QAT/MLA savings
- With differential attention: ensures the shared V has high rank -> better signal quality

### Estimated Improvement: -0.010 to -0.020 BPB

---

## 6. Engram / Register Memory Tokens

### Core Idea
Add learnable "memory" tokens that persist across layers, creating a cross-layer communication channel.

### Key Findings
- **Low priority** — the model already has full 1024-token causal attention (no context gap to fill)
- Existing x0 residual mixing + U-Net skips already provide cross-layer communication
- Custom attention masks break `is_causal=True` -> lose Flash Attention
- Two-pass workaround (separate cross-attention to memory) adds ~3-4% compute
- **Expected gain: only 0.001-0.005 BPB** — not worth the complexity

### Verdict
Skip for now. Would be valuable with sliding-window attention or much longer sequences.

---

## 7. Compression-Friendly Mixture of Experts (MoE)

### Core Idea
Shared base MLP + low-rank expert perturbations. Expert deltas are near-zero and compress to almost nothing, giving "free" capacity.

### Architecture
```python
# Each expert = shared_fc + delta_A[i] @ delta_B[i]  (rank-48 perturbation)
# 4 experts, soft routing (all contribute, weighted by learned gate)
# L1 regularization on deltas -> most stay near 0 -> INT5 values are {-1,0,1}
```

### Key Findings
- **29.2M total params but only ~11.1MB compressed** (vs 15.9MB for SOTA)
- ~5MB headroom for more experts, higher rank, or extra layers
- ~7% compute overhead (16% extra MLP FLOPs, MLP is ~45% of compute)
- Expert deltas compress 3-5x better than base weights (low-rank + L1 regularization)
- Soft routing avoids load-balancing loss waste

### Compression Analysis
```
SOTA (10L, 3x MLP, INT5):           ~15.9MB compressed
MoE (10L, 3x MLP + rank-48 deltas): ~11.1MB compressed  (saves ~4.8MB!)
```

### Estimated Improvement: -0.010 to -0.025 BPB

---

## Combined Architecture: The Recommended Design

### Configuration

```
Architecture:
  num_layers:      11  (funded by MLA param savings + INT5/6 QAT)
  model_dim:       512
  num_heads:       8
  attention:       MLA (latent_dim=128, content_dim=48, rope_dim=16)
  mlp_mult:        3   (hidden=1536)
  activation:      ReLU^2
  vocab:           1024 (tied embeddings)
  seq_len:         1024 (or 2048 if compute allows)
  skip:            U-Net (5 encoder + 6 decoder)

Additions:
  value_residual:  yes (per-head alpha, 40 params)
  smeargate:       yes (bigram blending)
  bigram_hash:     10240 buckets, dim=128
  logit_softcap:   30.0

Quantization:
  MLP weights:     INT5 QAT (STE during training)
  Attention:       INT6 QAT
  MLA small mats:  FP16 passthrough (< 65K params each)
  Embedding:       FP16 passthrough
  Compression:     zstd level 22

Training:
  optimizer:       Muon (matrices, lr=0.02, WD=0.04) + Adam (scalars/embeds)
  muon_momentum:   0.92 -> 0.99 over 1500 steps
  qat_enable:      step 300
  warmdown:        last 3000 steps (linear LR -> 0)
  swa:             every 50 steps, start_frac=0.4
  grad_clip:       0.3
```

### Parameter Budget

```
Token embedding (tied):          524,288
SmearGate:                           512
BigramHash (10240x128 + 128x512): 1,376,257
11x MLA Attention:               7,208,960  (655,360/layer)
11x MLP 3x:                    17,301,504  (1,572,864/layer)
11x Control (scales, norms):        22,528
Skip weights (5x512):               2,560
Value residual alphas:                  40
─────────────────────────────────────────
TOTAL:                         ~26,436,649 params
Estimated compressed size:     ~15.8 - 16.0 MB  ✓
```

### Interaction Matrix (Compounding vs Conflicting)

```
                    MLA   VR    QAT   SWA   SmearG  BigHash  11L
MLA                  --   +++    ++    +      =       =      +++
Value Residual      +++    --    +     +      =       =      +++
QAT (INT5/6)         ++    +    --    ++      =       =       ++
SWA                   +    +    ++    --      =       =        +
SmearGate             =    =     =     =     --      ++        =
BigramHash            =    =     =     =     ++      --        =
11 Layers           +++  +++    ++     +      =       =       --

Legend: +++ strong compound, ++ moderate, + weak, = orthogonal, - conflict
```

Key synergies:
- **MLA + 11 Layers**: MLA saves params that directly fund the 11th layer
- **Value Residual + 11 Layers**: VR prevents collapse that would otherwise limit depth gains
- **QAT + SWA**: QAT trains quantization-robust weights; SWA smooths distributions further

### 5-Phase Implementation Plan

```
Phase 1 (Day 1): Proven Foundation
  Fork best submission, add 11th layer, INT5/INT6 QAT, BigramHash 10240
  Expected: ~1.140 BPB, artifact <16MB

Phase 2 (Day 2): MLA
  Replace GQA with Multi-Latent Attention
  Expected: ~1.135 BPB

Phase 3 (Day 3): Value Residual
  Add V_0 blending across all layers
  Expected: ~1.128 BPB

Phase 4 (Day 4): Polish
  Hyperparameter sweep (LR, WD, warmdown, SWA params)
  10-seed validation for statistical significance
  Expected: ~1.125 BPB

Phase 5 (Day 5): Optional High-Risk
  Differential Attention (if custom Triton kernel feasible)
  Low-Rank MoE (if Phase 4 leaves compressed size headroom)
  Expected: ~1.118-1.125 BPB
```

### BPB Projection

```
Starting point (10L INT5 SOTA):     1.1428
+ 11th layer (QAT funded):        -0.003  -> 1.140
+ MLA replacing GQA:              -0.005  -> 1.135
+ Value Residual:                  -0.005  -> 1.130
+ Hyperparameter co-optimization:  -0.003  -> 1.127
+ (Optional) Diff Attn/MoE:       -0.005  -> 1.122

Confidence interval: 1.125 ± 0.010
Pessimistic (proven techniques only): 1.138
Optimistic (all compound as expected): 1.115
```

### Fallback Plan
If combined model doesn't converge:
1. **Tier 1**: Drop MLA and Diff Attn, keep 11L + VR + SmearGate + BigramHash + QAT -> ~1.135
2. **Tier 2**: Revert to proven 10L SOTA config + hyperparam tuning -> ~1.142
