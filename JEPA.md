# JEPA for Parameter Golf

> *"I'll send merch to anyone that can get a JEPA model to beat the parameter golf baseline! Only rule is no tokenizer (use byte level) to be true to JEPA."*

**Baseline to beat:** 1.2244 BPB (post-quantization, int8+zlib, 16MB budget, 10-min training on 8xH100)

**Constraints:** Byte-level (no tokenizer), JEPA-style architecture, must output valid cross-entropy loss for BPB scoring.

---

## The Core Challenge: BPB from a JEPA

JEPA predicts in **embedding space**, not token/byte space. But the challenge eval requires:
```python
def forward(self, input_ids, target_ids) -> scalar_cross_entropy_loss
```

This means we need a path from latent predictions to per-byte probabilities. Three approaches:

### Approach A: Hybrid JEPA + Generative Head (LLM-JEPA style)
Keep a standard cross-entropy head for BPB scoring, but add a JEPA objective to improve representations.
- **Loss:** `L = L_CE + λ * L_JEPA` (dual objective, as in [LLM-JEPA](https://arxiv.org/abs/2509.14252))
- **L_CE:** Standard next-byte cross-entropy (256-way softmax)
- **L_JEPA:** Cosine similarity or L2 between predicted embedding and EMA target embedding
- **At eval:** Only L_CE matters for BPB scoring; JEPA loss is auxiliary
- **Why it helps:** JEPA forces learning abstract representations rather than surface statistics. LLM-JEPA shows this reduces overfitting and improves across model families
- **Feasibility:** HIGH — minimal code changes, keeps standard eval path

### Approach B: Energy-Based Scoring
Use JEPA energy function directly to score byte sequences.
- **Energy:** `E(x, y) = ||Predictor(Encoder(x)) - TargetEncoder(y)||²`
- **Convert to probability:** `p(y_t | x) ∝ exp(-E(x, y_t))` over all 256 byte values
- **BPB:** `-log2(p(y_t)) / n_bytes`
- **Problem:** Requires normalizing over 256 candidates per position — 256 forward passes per byte, or clever batching. Feasible but ~256x slower eval
- **Feasibility:** MEDIUM — correct in principle but computationally expensive

### Approach C: JEPA Encoder + Lightweight Decoder
Pure JEPA encoder learns representations, thin decoder maps to byte probabilities.
- JEPA encoder: predicts future embeddings from context (self-supervised)
- Decoder: single linear layer `embedding_dim → 256` for byte prediction
- Two-stage: pretrain JEPA, then train decoder (or joint training)
- Similar to [VL-JEPA](https://arxiv.org/abs/2512.10942) where decoder is invoked only when needed
- **Feasibility:** HIGH — clean separation, easy to evaluate

---

## Recommended Architecture: Byte-Level JEPA-LM

### Overview
A byte-level autoregressive transformer with an auxiliary JEPA prediction objective. The JEPA component forces the model to learn abstract, compressible representations while the CE head provides exact BPB scoring.

### Architecture Details

```
Input: raw bytes (vocab_size=256, no tokenizer)
  ↓
Byte Embedding (256 × d_model) — tiny table, ~50KB at d=192
  ↓
Positional Encoding (RoPE or learned)
  ↓
N × Transformer Blocks (causal attention + MLP)
  │
  ├─→ CE Head: Linear(d_model, 256) → cross_entropy(logits, target_bytes)
  │   [Standard autoregressive loss for BPB scoring]
  │
  └─→ JEPA Head: Predictor(context_emb) → predicted_target_emb
      │   [Predicts embedding of masked/future byte spans]
      │
      └── Target: EMA_Encoder(target_bytes) → target_emb
          [Stop-gradient, exponential moving average of encoder]
```

### JEPA Component Design

**Masking strategy (adapted from I-JEPA / LeWM):**
- Context: bytes 0..t (causal, same as standard LM)
- Target: bytes t+1..t+k (next k bytes, k=8..64)
- Predictor: lightweight transformer (2-3 layers) that takes context embedding and predicts target span embedding
- Target encoder: EMA copy of main encoder (momentum 0.996→0.999 cosine schedule)

**JEPA Loss:**
- L_JEPA = 1 - cos_sim(Predictor(h_context), sg(EMA_Encoder(target_span)))
- Or: L_JEPA = ||Predictor(h_context) - sg(EMA_Encoder(target_span))||²
- `sg()` = stop gradient

**Collapse prevention (critical for JEPA):**
Options ranked by practicality:
1. **EMA target encoder** — standard in I-JEPA/V-JEPA, simplest, proven. Momentum 0.996+
2. **VICReg regularization** — variance + covariance terms on embeddings. Proven in [C-JEPA](https://proceedings.neurips.cc/paper_files/paper/2024/file/04a80267ad46fc730011f8760f265054-Paper-Conference.pdf). Add `L_var + L_cov` to loss
3. **SIGReg** (from [LeWM](https://le-wm.github.io)) — enforce Gaussian latent distribution via Cramér-Wold projections. Only 1 hyperparameter (λ). Used in the most recent JEPA world model work (March 2026)

**Recommended: EMA + VICReg** (simple, proven, low overhead)

### Byte-Level Challenges

**Problem:** Byte sequences are ~3-4x longer than token sequences for same text.
- 1024 tokens ≈ 3000-4000 bytes
- Quadratic attention on 4K bytes is expensive
- 10-min training budget is tight

**Mitigations:**
1. **Local attention + sparse global:** Sliding window (256 bytes) + strided global attention every 64 bytes
2. **Byte patching** (MegaByte-style): Group bytes into patches of 4-8, process patches with global transformer, local transformer within patches. Reduces sequence length 4-8x
3. **Linear attention** for some layers: Flash-linear-attention for O(n) complexity
4. **Shorter training sequences:** Train on 512-1024 byte sequences (shorter but more steps)

### Concrete Model Sizing (16MB budget)

```
Byte-level, no tokenizer overhead:
  Embedding: 256 × 384 = 98K params (negligible)
  Transformer: 12 layers, d=384, 6 heads, MLP 4x
    Per layer: ~1.2M params
    Total: ~14.4M params
  CE Head: 384 × 256 = 98K params (or tie with embedding)
  JEPA Predictor: 2 layers, d=384 = ~2.4M params
    (predictor can be discarded at eval if not needed)

  Total trainable: ~17M params
  Total in artifact: ~14.6M params (no predictor at eval)
  Int4 + zlib: ~14.6M × 0.5 / 3 ≈ 2.4MB → massive headroom

  With headroom, scale to:
  24 layers, d=512, 8 heads → ~40M params → ~6-7MB artifact
```

**Key insight:** Byte-level models have tiny embedding tables (256 entries vs 1024-4096 for tokenizers), so almost all parameters go into transformer layers. This is parameter-efficient.

### Training Schedule (10 minutes, 8xH100)

```
Phase 1 — Joint JEPA + CE training (7 min):
  Loss = L_CE + 0.5 * L_JEPA + 0.01 * (L_var + L_cov)
  LR: 3e-4 with warmup 500 steps
  Batch: 256 sequences × 1024 bytes = 262K bytes/step
  ~8K steps on 8xH100 with FP8

Phase 2 — CE-only fine-tuning (2 min):
  Loss = L_CE only
  LR: 1e-4 → 1e-5 cosine decay
  ~2.5K steps
  Purpose: maximize BPB without JEPA interference

Phase 3 — Quantize + compress (1 min):
  Int4 group quantization (g=128)
  STE-based QAT if time permits
  zlib compression
```

### Discard predictor at eval:
- JEPA predictor + EMA encoder are training-only
- Final artifact = encoder + CE head only
- This saves ~15-20% of parameters

---

## Why JEPA Might Actually Help at Byte Level

### 1. Abstract representation learning
Byte-level models waste capacity learning surface patterns (UTF-8 encoding, HTML tags, punctuation patterns). JEPA forces the encoder to build abstract representations that capture semantics, not just byte co-occurrence. This is exactly why I-JEPA outperforms MAE in vision — predicting in embedding space ignores unpredictable pixel-level noise.

### 2. Multi-scale prediction
Standard next-byte CE predicts one byte at a time. JEPA predicts embeddings of future *spans*, forcing the model to capture longer-range dependencies. This is analogous to predicting "concepts" rather than "pixels" — the model must understand what comes next at a semantic level.

### 3. Regularization / anti-overfitting
LLM-JEPA shows that adding the JEPA objective makes fine-tuning robust to overfitting. For our 10-min training budget with ~40M params, overfitting is a real risk. The JEPA term acts as a powerful regularizer that encourages learning transferable features.

### 4. Better compression
JEPA-trained representations tend to be smoother and more Gaussian (especially with VICReg/SIGReg). Smoother weight distributions compress better under int4+zlib.

---

## Why JEPA Might NOT Work (Honest Assessment)

### 1. No existing byte-level JEPA baseline
This is completely uncharted territory. No one has published a byte-level JEPA language model. The combination of byte-level + JEPA + small model + BPB eval is novel — which means high risk.

### 2. BPB eval only uses CE head
The JEPA component is purely auxiliary during eval. If it doesn't meaningfully improve the CE head's predictions, it's wasted training compute. The 10-min budget is tight — every training step spent on JEPA loss is a step not spent on pure CE optimization.

### 3. Collapse risk in short training
JEPA collapse prevention (EMA, VICReg) needs enough training steps to stabilize. With only ~10K steps, the JEPA component might not converge properly, or worse, collapse and poison the shared encoder.

### 4. Byte sequences are long
At 3-4x the length of token sequences, byte-level models need more compute per training step. The JEPA predictor adds another ~20% overhead. Combined, we might get 40-50% fewer effective training steps compared to a tokenized baseline.

### 5. The baseline is strong
1.2244 BPB with a well-tuned tokenized model is hard to beat. Byte-level models typically need ~3-10x more compute to match tokenized models at the same parameter count (EvaByte, MegaByte findings). Our 10-min budget doesn't provide that margin.

---

## Minimum Viable Experiment

Before committing to full implementation, test the core hypothesis:

### Step 1: Byte-level baseline (no JEPA)
- Standard autoregressive byte-level transformer
- 12 layers, d=384, 6 heads
- Train 10 min, measure BPB
- **Expected:** ~1.35-1.50 BPB (worse than tokenized baseline due to longer sequences)

### Step 2: Add JEPA objective
- Same architecture + JEPA predictor + EMA target encoder
- Loss = L_CE + 0.5 * L_JEPA
- Train 10 min, measure BPB
- **Target:** < 1.30 BPB (JEPA must recover >50% of the byte-level penalty)

### Step 3: Scale up if promising
- Larger model (leverage byte-level embedding savings)
- MegaByte-style patching for efficiency
- Int4 quantization for maximum params in 16MB
- **Target:** < 1.2244 BPB (beat baseline)

---

## Related Work & References

### JEPA Core
- [I-JEPA (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf) — Original image JEPA
- [V-JEPA 2 (2025)](https://arxiv.org/abs/2506.09985) — Video JEPA at scale
- [LeWorldModel (March 2026)](https://le-wm.github.io) — Stable end-to-end JEPA from pixels with SIGReg. 15M params, single GPU. Code: [github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)

### JEPA + Language
- [LLM-JEPA (Sep 2025)](https://arxiv.org/abs/2509.14252) — Dual CE+JEPA objective for LLMs. Outperforms standard training, resistant to overfitting. Code: [github.com/galilai-group/llm-jepa](https://github.com/galilai-group/llm-jepa)
- [VL-JEPA (Dec 2025)](https://arxiv.org/abs/2512.10942) — Vision-language JEPA, 50% fewer trainable params than classical VLMs
- [LANG-JEPA](https://github.com/jerber/lang-jepa) — Experimental text JEPA predicting semantic features of future text
- [VICReg for Sentence Embedding](https://github.com/domenicrosati/vicreg-for-sentence-embedding) — JEPA + VICReg loss for language

### Collapse Prevention
- [VICReg (ICLR 2022)](https://arxiv.org/abs/2105.04906) — Variance-Invariance-Covariance regularization
- [C-JEPA (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/04a80267ad46fc730011f8760f265054-Paper-Conference.pdf) — Contrastive JEPA with VICReg integration
- LeWM SIGReg — Cramér-Wold Gaussian enforcement, single hyperparameter

### Byte-Level LMs
- [EvaByte (2025)](https://hkunlp.github.io/blog/2025/evabyte/) — Efficient byte-level LMs at scale
- MegaByte — Byte patching for efficient byte-level transformers
- [Bridging the Gap for Tokenizer-Free LMs](https://arxiv.org/pdf/1908.10322) — Early work on byte-level challenges

### Energy-Based Scoring
- [Residual EBMs for Text (ICLR)](https://openreview.net/pdf?id=B1l4SgHKDH) — Joint LM + energy model, lower perplexity with fewer params
- [Energy-Based Diffusion LMs (2024)](https://arxiv.org/html/2410.21357v1) — EDLM with BPC evaluation
- [Continuous AR LMs / BrierLM](https://arxiv.org/pdf/2510.27688) — Likelihood-free evaluation for continuous models
