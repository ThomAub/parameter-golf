# Test-Time Compute for Parameter Golf

## Challenge Context

The Parameter Golf rules **explicitly encourage** test-time compute (TTC):

> "We won't accept submissions that take more than **10 minutes on 8xH100** to evaluate, but otherwise you're free to evaluate however."
> "We encourage competitors to push the bounds of evaluation methods as aggressively as with training methods."

The README even lists "test-time compute" as an expected technique alongside "aggressive parameter tying" and "depth recurrence."

**Key insight**: The current baseline evaluates in ~1-2 seconds. We have up to 10 minutes of eval time on 8xH100s — that's ~300-600x more compute available for inference. This is a massive untapped resource.

---

## Technique 1: Sliding Window Evaluation

### What it does
The current eval processes the validation set as **non-overlapping** 1024-token chunks. The first token in each chunk has **zero** prior context, the second has 1 token of context, etc. On average, tokens only have ~512 tokens of context.

With a **sliding window** (stride < seq_len), we process overlapping windows and only score tokens that have sufficient context. For example, with window=1024 and stride=512, every scored token has at least 512 tokens of prior context.

### Expected BPB improvement
- **Stride = seq_len/2**: ~0.01–0.03 BPB improvement
- **Stride = 1** (maximum): ~0.02–0.05 BPB (but extremely expensive)
- Sweet spot: stride=128–256 balances quality and compute

### Compute cost
With window W and stride S, compute scales as W/S relative to baseline:
- stride=512 (W=1024): 2x compute
- stride=256: 4x compute
- stride=128: 8x compute
- stride=64: 16x compute

With 10 minutes of eval budget and current eval taking ~2 seconds, we can afford stride≈2 (essentially stride=1 recomputing everything), but memory is the practical constraint.

### Implementation

Current eval code (`eval_val`, line 219):
```python
# Current: non-overlapping chunks
for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
    x = local[:-1].reshape(-1, args.train_seq_len)
    y = local[1:].reshape(-1, args.train_seq_len)
    batch_loss = model(x, y).detach()
```

Modified sliding window:
```python
# New: overlapping windows, only score last `stride` tokens per window
eval_seq_len = args.train_seq_len  # or longer (see Technique 3)
stride = eval_seq_len // 4  # tune this

# Process validation tokens with overlapping windows
pos = 0
while pos + eval_seq_len < total_val_tokens:
    window = val_tokens[pos : pos + eval_seq_len + 1]
    x = window[:-1].unsqueeze(0)
    y = window[1:].unsqueeze(0)

    # Forward pass gets logits for all positions
    logits = model.forward_logits(x)  # need to modify model to return logits

    # Only score the last `stride` tokens (which have full context)
    score_start = eval_seq_len - stride
    scored_logits = logits[:, score_start:, :]
    scored_targets = y[:, score_start:]

    # Accumulate loss and byte counts
    loss = F.cross_entropy(scored_logits.reshape(-1, V), scored_targets.reshape(-1), reduction='sum')
    ...

    pos += stride
```

**Key change needed**: The current `model.forward()` returns loss directly. We need a `forward_logits()` variant that returns raw logits so we can score only the tail tokens.

### Interaction with BPB calculation
The BPB calculation uses lookup tables for byte counting. With sliding windows, we need to track which tokens we're scoring and apply the byte accounting only to those scored tokens. The `has_leading_space_lut` and `is_boundary_token_lut` logic needs the previous token for each scored token, which is available since we process full windows.

---

## Technique 2: Depth Recurrence (Universal Transformer)

### What it does
Instead of N unique transformer layers, use K < N unique layers and **loop** them multiple times:
- **Training**: K unique layers × L_train loops = K*L_train effective layers
- **Eval**: K unique layers × L_eval loops = K*L_eval effective layers (L_eval > L_train)

Same parameter count at any loop count, but more effective depth at eval. The model learns **iterative refinement** — each pass through the shared layers further refines the hidden representations.

### Architecture design

**Option A: Full weight sharing (K=1 unique block set)**
- 1 set of attention+MLP weights, looped 9x during training, 18x+ during eval
- Most aggressive compression but hardest to train (all layers must serve all depths)
- Parameter savings: 9x fewer layer params → can make each layer much wider

**Option B: Partial sharing (K=3 unique block sets)**
- 3 unique layers looped 3x each during training (= 9 effective layers)
- At eval: loop 6x each (= 18 effective layers) or more
- Good balance of expressivity and parameter efficiency

**Option C: Grouped sharing with U-Net adaptation**
- Keep the U-Net skip connection structure
- Share weights within encoder group and decoder group separately
- Most compatible with existing architecture

### Recommended: Option B (K=3, L_train=3)

```
Training forward pass (9 effective layers):
  Block_0 → Block_1 → Block_2 → Block_0 → Block_1 → Block_2 → Block_0 → Block_1 → Block_2

Eval forward pass (18 effective layers):
  Block_0 → Block_1 → Block_2 → ... (repeated 6 times)
```

**Parameter budget impact** (with K=3 instead of K=9):
- Current: 9 layers × 1.84M params/layer = 16.5M layer params + 0.5M embedding = 17.1M total
- With K=3: 3 layers × 1.84M = 5.5M layer params + 0.5M = 6.0M total
- **Freed budget: ~11M params** → can widen model (d=768 or d=896) or add more experts

### Key implementation considerations

1. **Position-aware looping**: Each loop iteration should know which iteration it's in. Options:
   - Add a learnable "loop embedding" that's added to the hidden state at each loop start
   - Use different RoPE bases per loop iteration
   - Simply let the model learn to handle iteration implicitly

2. **Skip connections with recurrence**: The U-Net skip structure needs adaptation:
   - Option: Only apply skip connections in the last loop pass
   - Option: Store skips from first pass, apply in last pass
   - Option: Drop U-Net skips entirely (recurrence provides similar benefits)

3. **Gradient flow**: With depth recurrence, gradients flow through K*L_train layers of shared weights. Need:
   - Careful learning rate tuning (lower than baseline)
   - Possibly gradient checkpointing for memory
   - Consider not backpropagating through all loops (truncated BPTT)

4. **Convergence during training**: Depth-recurrent models can be harder to train:
   - Start with L_train=2 and increase (curriculum)
   - Or use input injection (re-add x0 at each loop start, already done via `resid_mix`)

### Expected BPB improvement
- With same parameter budget but 2x eval depth: ~0.02–0.05 BPB
- With freed parameters invested in width: ~0.05–0.10 BPB
- **Combined (wider + deeper at eval)**: potentially 0.05–0.15 BPB

This is the highest-leverage TTC technique because it's both a parameter efficiency win (fewer unique layers = more budget for width) AND a test-time compute win (more loops at eval).

### References
- "Universal Transformers" (Dehghani et al., 2019) — original depth recurrence
- "Adaptive Computation Time" (Graves, 2016) — dynamic halting per token
- "Think before you speak" (meta-internal) — iterative refinement
- PonderNet (Banino et al., 2021) — learned halting with geometric priors
- Block Recurrence Transformer (Hutchins et al., 2022) — recurrence with limited context

---

## Technique 3: Longer Context at Evaluation

### What it does
Train with seq_len=1024 but evaluate with seq_len=2048, 4096, or even 8192. More tokens get conditioned on longer history, reducing BPB on average. The challenge explicitly allows this:

> "As with modded-nanogpt, we allow evaluation at any sequence length."

### RoPE extrapolation challenge
Standard RoPE degrades significantly beyond training length. Solutions:

**A. NTK-aware RoPE scaling (simplest)**
Scale the base frequency: `base_new = base * (eval_len / train_len) ^ (dim / (dim-2))`
- For 1024→2048: base_new ≈ 20,000 (from 10,000)
- For 1024→4096: base_new ≈ 40,000
- Works reasonably well up to 2-4x training length

**B. YaRN (Yet another RoPE extensioN)**
Combines NTK scaling with attention scaling and temperature adjustment:
- `sqrt(1/s)` attention scaling where s = eval_len/train_len
- Frequency-dependent interpolation (low frequencies interpolated, high frequencies kept)
- Better than NTK alone for large extrapolation ratios

**C. Dynamic NTK (simplest, good enough)**
Only scale when position > train_len: for positions within training length, use original RoPE.

### Implementation
Modify the `Rotary` module:
```python
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024):
        super().__init__()
        self.base = base
        self.dim = dim
        self.train_seq_len = train_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        # Dynamic NTK scaling for lengths beyond training
        if seq_len > self.train_seq_len:
            ratio = seq_len / self.train_seq_len
            new_base = self.base * ratio ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (new_base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        else:
            inv_freq = self.inv_freq.to(device)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)
```

### Expected BPB improvement
- 1024→2048: ~0.005–0.015 BPB
- 1024→4096: ~0.010–0.025 BPB (but quality degrades if extrapolation fails)
- Best combined with sliding window at the longer context length

### Compute cost
- 2048 context: ~4x per-token compute (quadratic attention)
- 4096 context: ~16x per-token compute
- With Flash Attention, memory is O(n) but compute is still O(n²)
- Still easily fits in 10-minute eval budget

### Risk
RoPE extrapolation quality is uncertain without fine-tuning. Models trained at 1024 may not generalize well to 4096. Safe bet is 2048 with NTK scaling. Could also:
- Train at 512 and eval at 1024+ (model sees "normal" 1024 within extrapolation range)
- Use ALiBi instead of RoPE (better extrapolation, but needs architecture change)

---

## Technique 4: Checkpoint / Depth Ensemble

### Option A: Checkpoint Ensemble (weight space)
Average weights from multiple training checkpoints ("model soup"). This is done **before** saving, so no extra eval compute. Not strictly TTC but free improvement.

- Save checkpoints at steps N-2000, N-1000, N-500, N (during warmdown)
- Average their weights (uniform or exponential weighting)
- Expected gain: ~0.002–0.005 BPB (modest but free)

### Option B: EMA (Exponential Moving Average)
Maintain an EMA of weights during training with decay ~0.999–0.9999.
- EMA weights are often better than final weights
- One extra copy of weights during training (memory cost)
- Save EMA weights as the submission
- Expected gain: ~0.003–0.008 BPB

### Option C: Depth Ensemble (with depth recurrence)
If we implement depth recurrence (Technique 2), we can ensemble predictions from different loop counts:
- Run model with L=6 loops → get logits_6
- Run model with L=9 loops → get logits_9
- Run model with L=12 loops → get logits_12
- Average: logits_final = (logits_6 + logits_9 + logits_12) / 3

**This is true test-time compute** — same model, multiple forward passes at different depths, averaged predictions. Each depth provides a different "view" of the data.

Cost: 3x inference compute (still trivial within 10-minute budget).
Expected gain: ~0.005–0.015 BPB on top of depth recurrence alone.

### Option D: Stochastic Depth Ensemble
During eval, run multiple forward passes with different randomly dropped layers:
- Pass 1: all layers active
- Pass 2: drop layer 3, 7
- Pass 3: drop layer 1, 5
- Average logits across passes
- Expected gain: small (~0.002–0.005 BPB) but nearly free

### Recommended: B + C (EMA during training + depth ensemble at eval)

---

## Combined Strategy: Maximum TTC

### Architecture changes (training time):
1. **Depth recurrence**: K=3 unique layers, L_train=3 (9 effective layers)
2. **Use freed parameters for width**: d=512→d=700+ (or add MoE experts)
3. **EMA weights**: maintain EMA during training with decay=0.9995

### Evaluation pipeline (inference time):
1. **Load EMA weights** from compressed artifact
2. **Dynamic NTK RoPE scaling** for eval_seq_len=2048
3. **Sliding window** with stride=128, window=2048
4. For each window:
   a. Run forward pass with **L=4 loops** → logits_4
   b. Run forward pass with **L=6 loops** → logits_6
   c. Run forward pass with **L=8 loops** → logits_8
   d. Average logits: (logits_4 + logits_6 + logits_8) / 3
5. Score only the last `stride` tokens per window
6. Accumulate BPB across all windows

### Estimated compute budget
- Val tokens: ~50M tokens in validation set
- With stride=128, window=2048: ~390K windows
- 3 depth ensemble passes per window: 1.17M forward passes
- Each forward pass: 2048 tokens through ~24 effective layers
- H100 at ~1000 TFLOPS bf16: this should complete in ~2-5 minutes on 8xH100

### Estimated total BPB improvement
| Technique | Estimated gain | Compute cost |
|---|---|---|
| Sliding window (stride=128, W=2048) | 0.015–0.030 | 16x |
| Depth recurrence (3→8 loops at eval) | 0.03–0.08 | 2.7x |
| Longer context (1024→2048 + NTK) | 0.005–0.015 | 4x |
| Depth ensemble (3 depths) | 0.005–0.015 | 3x |
| EMA weights | 0.003–0.008 | 1x (free) |
| **Width increase from freed params** | **0.03–0.06** | 1x |
| **Total estimated** | **0.09–0.21** | ~130x |

Starting from baseline 1.2244, this could reach **1.01–1.13 BPB** — a massive improvement.

---

## Implementation Priority

1. **Sliding window eval** — Easiest, guaranteed improvement, no architecture change
2. **Depth recurrence** — Highest potential, requires architecture redesign
3. **EMA weights** — Simple training change, small but free gain
4. **Longer eval context + NTK RoPE** — Moderate effort, good with sliding window
5. **Depth ensemble** — Falls out naturally from depth recurrence

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Depth recurrence hurts training convergence | High | Curriculum learning (start L=2, increase to L=3). Use input injection (x0 residual already present). |
| RoPE extrapolation fails beyond 2048 | Medium | Stick to 2x extrapolation (1024→2048). Test with NTK scaling before committing. |
| Sliding window eval is too slow | Low | Use batched windows. 8xH100 has plenty of memory and compute. |
| INT8 quantization of shared weights hurts more with recurrence | Medium | Shared weights are used at every loop, so quantization error compounds. May need INT8 QAT or careful calibration. |
| Model doesn't benefit from extra loops at eval | Medium | Monitor eval loss vs loop count. If diminishing returns plateau at L=4, stop there. |

---

## Analysis of PR #457: State-of-the-Art Submission (1.1839 BPB)

PR #457 by carlesonielfa is the strongest submission seen so far, achieving **1.1839 BPB** (vs 1.2244 baseline). Here's what it does and what we can learn.

### Results Breakdown
- **Without TTT**: 1.2192 BPB (int8+zlib) — already beats baseline by 0.005
- **With cross-doc TTT**: 1.1839 BPB — a further **0.035 BPB** from test-time training alone
- Model size: 15.35 MB, training: 13,137 steps / 600s on 8xH100

### Technique Analysis

#### 1. seq_len=4096 (Biggest training improvement)
Simply increased training sequence length from 1024 to 4096. Per the author, this is "the single largest contributor." This gives the model more context during training, which directly improves sliding window eval. No RoPE tricks needed since the model natively trains at 4096.

**Lesson for us**: Instead of training at 1024 and extrapolating with NTK scaling, just train at 4096 directly. The 4x longer sequences mean 4x fewer sequences per batch at same token count, which is fine. Flash Attention makes the memory overhead manageable.

#### 2. Exclusive Self-Attention (XSA) — Applied to deepest 4 layers
After SDPA, subtracts from each head's output the component aligned with its value vector:
```python
v_norm = F.normalize(v.float(), dim=-1).to(y.dtype)
dot = (y * v_norm).sum(-1, keepdim=True)
y = y - dot * v_norm  # Remove value-aligned component
```
**Intuition**: Forces attention to capture information NOT already in the value vectors. This is a form of information decorrelation — the attention output carries only novel signal, not redundant value content. Applied to deep layers where value representations are most refined.

**Cost**: ~512 parameters (none extra — it's a structural change, not a parameter addition). The normalization and projection add minimal compute.

#### 3. Value Residual Learning (VRL) — Per-layer learnable residual from layer 0
Each layer i > 0 gets: `v_i = v_i + lambda_i * v_first` where `v_first` is layer 0's value output and `lambda_i` is a learnable scalar initialized to 0.

**Intuition**: Provides a "ground truth" reference signal at every depth. Deep layers can directly access the raw token representations through value vectors, preventing representation collapse. Similar in spirit to DenseNet skip connections but targeted at the value stream.

**Cost**: 10 scalars (one per non-first layer). Essentially free.

#### 4. SmearGate — Learned token blending at embedding
```python
g = sigmoid(gate)  # 512 params, init to 0 → sigmoid(0) = 0.5
output = (1 - g) * x + g * x_prev  # blend current with previous token
```
**Intuition**: A lightweight bigram prior. Tokens carry information about their predecessor, giving the model sub-token context even at position 0. Initialized to 50% blend.

**Cost**: 512 parameters.

#### 5. Stochastic Weight Averaging (SWA) — 24 checkpoints from last 40% of warmdown
Collects 24 evenly-spaced checkpoints during the final 40% of warmdown, then averages all weights at the end.

**This is our "checkpoint ensemble" Technique 4A, but better tuned**: 24 checkpoints (not 2-3), from the warmdown phase specifically (where the model is settling). The result is a flatter-minimum model that generalizes better.

**Cost**: Memory during training to store 24 state dicts. No extra eval compute.

#### 6. Cross-Document Test-Time Training (TTT) — The killer feature (+0.035 BPB!)
This is the **most impactful TTC technique** in the PR, providing a **0.035 BPB** improvement on top of an already-strong model.

**How it works:**
1. Rank-8 LoRA adapters are added to Q, V projections and the LM head
2. For each document in the validation set:
   - Process the document in chunks using a sliding window (stride=64)
   - For each chunk, compute the model's predictions (scoring)
   - After scoring, **train the LoRA adapters** on the already-scored tokens
   - The next chunk benefits from the adapted model
3. Cross-document variant: the context window extends before the document boundary, giving the base model cross-document context while training LoRA only on in-document tokens

**Why it works:**
- Each document has its own distributional characteristics (topic, style, vocabulary)
- LoRA with rank 8 has very few parameters (~50K) so it can adapt quickly
- By training on already-scored tokens, there's no data leakage
- The base model provides strong priors; LoRA just does fine-grained domain adaptation

**Implementation details from the PR:**
- `BatchedLinearLoRA`: maintains independent LoRA weights per batch element (for multi-document parallelism)
- Optimizer state is reset between document batches
- LoRA targets: Q projection, V projection, LM head
- VRL also gets LoRA adapters with `vrl_delta` (value residual also adapts per document)

#### 7. Warmdown-QAT
Not explicit QAT — instead, the extended warmdown (1200 iters) with SWA produces weights that quantize well. The quant penalty is only **+0.0009 BPB** (1.2183 pre-quant → 1.2192 post-quant), which is remarkable.

**Lesson**: SWA inherently produces smoother weight distributions that compress better. No need for explicit QAT if warmdown + SWA is done well.

### What We Should Adopt

| Technique | Priority | Effort | Expected Gain | Notes |
|---|---|---|---|---|
| **Cross-doc TTT** | **Critical** | High | **0.03-0.04 BPB** | The single biggest TTC win. Must implement. |
| **seq_len=4096 training** | **Critical** | Low | 0.02-0.04 BPB | Just change a hyperparameter (+ more data I/O) |
| **SWA (24 ckpts)** | High | Low | 0.003-0.008 BPB | Free quality + better quantization |
| **Sliding window eval (stride=64)** | High | Medium | 0.01-0.02 BPB | Required for TTT anyway |
| **XSA** | Medium | Low | 0.002-0.005 BPB | Simple code change on deep layers |
| **VRL** | Medium | Low | 0.002-0.005 BPB | 10 scalar params, easy win |
| **SmearGate** | Low | Low | 0.001-0.003 BPB | 512 params, marginal gain |

### Revised Strategy

Given PR #457's results, our priority order should be:

1. **seq_len=4096 training** — biggest training improvement, trivial change
2. **Sliding window eval** — required infrastructure for everything else
3. **Cross-document TTT with LoRA** — the killer TTC technique (+0.035 BPB!)
4. **SWA** — free improvement during training
5. **XSA + VRL** — small architectural wins, low effort
6. **Depth recurrence** — still valuable for parameter efficiency, but TTT may be more impactful per-unit-effort

The combination of seq_len=4096 + sliding window + cross-doc TTT + SWA should be our primary focus. Depth recurrence becomes a secondary optimization for freeing parameter budget.
