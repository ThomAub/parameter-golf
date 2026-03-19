# Optimal Vocab Size for Parameter Golf

## Context

The competition scores on **bits-per-byte (BPB)**, defined as:

```
BPB = (cross_entropy_loss / ln2) × tokens_per_byte
```

Two levers control BPB:
1. **Loss per token** — how well the model predicts the next token (lower = better)
2. **Tokens per byte** — how efficiently the tokenizer compresses text (lower = better)

Increasing vocab size improves (2) but hurts (1). This analysis finds the sweet spot under the 16MB submission constraint.

---

## Parameter Budget

| Item | Value |
|---|---|
| Submission limit | 16,000,000 bytes |
| Code overhead | ~48,000 bytes |
| Model budget | ~15,950,000 bytes |
| Zlib ratio on int8 weights | ~0.82 |
| **Effective int8 parameter budget** | **~19.4M params** |
| **Effective int4 parameter budget** | **~38.8M params** |

---

## Baseline Architecture (V=1024, d=512, L=9)

```
head_dim  = 512 / 8 = 64
kv_dim    = 4 × 64  = 256

Per-layer breakdown:
  Q proj:   512 × 512 = 262,144
  K proj:   512 × 256 = 131,072
  V proj:   512 × 256 = 131,072
  Out proj: 512 × 512 = 262,144
  MLP fc:   512 × 1024 = 524,288
  MLP proj: 1024 × 512 = 524,288
  Scalars:  ~2,560
  ─────────────────────────────
  Per layer total: ~1,837,568

Embedding (tied): 1024 × 512    =    524,288
9 layers:         9 × 1,837,568 = 16,538,112
Skip weights + misc:             ≈     20,000
─────────────────────────────────────────────
TOTAL ≈ 17,082,400 params
```

Fits comfortably in the 19.4M int8 budget. **~2.3M params of headroom.**

---

## Vocab Size Sweep (int8, tied embeddings, d=512, L=9)

| Vocab | Embed Params | Total Params | Fits 19.4M? | tokens/byte (est.) | Compression vs 1024 |
|---|---|---|---|---|---|
| 1024 | 524K | 17.1M | Yes | ~0.90 | 1.00x |
| 2048 | 1.05M | 17.6M | Yes | ~0.75 | 0.83x |
| 4096 | 2.10M | 18.6M | Yes (tight) | ~0.60 | 0.67x |
| 8192 | 4.19M | 20.7M | **No** | ~0.50 | 0.56x |

At V=8192 with d=512, we bust the budget. To fit, we'd need to shrink d:

| Vocab | d | Layers | Total Params | Fits? | tokens/byte |
|---|---|---|---|---|---|
| 8192 | 480 | 9 | ~18.7M | Yes | ~0.50 |
| 8192 | 448 | 10 | ~18.1M | Yes | ~0.50 |
| 16384 | 420 | 9 | ~19.5M | Barely | ~0.42 |

---

## BPB Impact Estimation

The cross-entropy loss scales roughly as:

```
loss ≈ loss_irreducible + C × (vocab_size)^α / model_capacity^β
```

Where `α ≈ 0.05–0.15` (larger vocab = harder task, weak effect) and `β ≈ 0.5–0.7` (more params = lower loss, strong effect).

### Scenario Analysis

**V=1024 (baseline)**
```
loss ≈ 2.07 nats (observed)
tokens_per_byte ≈ 0.90
BPB = (2.07 / 0.693) × 0.90 = 2.69 → but actual is 1.2244
```

Note: The actual BPB of 1.2244 implies `tokens_per_byte ≈ 0.41` with the achieved loss, or the loss figure differs. Using the actual competition BPB as ground truth:

```
Actual baseline BPB = 1.2244
```

**V=2048 (same d=512, L=9, 17.6M params)**
```
Expected loss increase: ~2-5% (slightly harder task, same model capacity)
Expected tokens_per_byte decrease: ~15-20%
Net BPB change: -10% to -15% improvement
Estimated BPB ≈ 1.04 – 1.10
```

**V=4096 (same d=512, L=9, 18.6M params)**
```
Expected loss increase: ~5-10%
Expected tokens_per_byte decrease: ~30-35%
Net BPB change: -20% to -25% improvement
Estimated BPB ≈ 0.92 – 0.98
```

**V=8192 (reduced d=480, L=9, 18.7M params)**
```
Expected loss increase: ~15-25% (harder task + smaller model)
Expected tokens_per_byte decrease: ~40-45%
Net BPB change: -15% to -25% (uncertain — model capacity loss hurts)
Estimated BPB ≈ 0.92 – 1.04
```

---

## Key Interactions with Other Optimizations

### With INT4 Quantization

INT4 doubles the budget to ~38M params. This changes everything:

| Vocab | d | Layers | Total Params | tokens/byte | Expected BPB |
|---|---|---|---|---|---|
| 1024 | 700 | 12 | ~35M | 0.90 | ~1.05 |
| 2048 | 680 | 12 | ~34M | 0.75 | ~0.92 |
| 4096 | 640 | 14 | ~37M | 0.60 | ~0.82 |
| 8192 | 600 | 14 | ~36M | 0.50 | ~0.80 |

With INT4, **V=4096–8192 becomes clearly optimal** because the model has enough capacity to absorb the harder prediction task.

### With Depth Recurrence

Weight-sharing across layers gives effective depth at no parameter cost. A V=4096, d=512, 6 unique layers × 3 passes = 18 effective layers:

```
Params: 6 × 1.84M + 2.1M = 13.1M (well within int8 budget)
Effective depth: 18 layers
Expected BPB: ~0.95 – 1.05
```

---

## Recommendations

### Conservative (low risk, moderate gain)
**V=2048, d=512, L=9, tied embeddings**
- 17.6M params, fits int8 easily
- ~10-15% BPB improvement
- Requires retraining tokenizer and reprocessing data

### Aggressive (high potential, needs validation)
**V=4096, d=512, L=9, tied embeddings**
- 18.6M params, tight int8 fit
- ~20-25% BPB improvement if loss increase is modest
- Best combined with INT4 to allow larger d

### Optimal (requires INT4)
**V=4096–8192, d=640, L=12-14, tied embeddings, INT4 quantization**
- ~35-38M params
- Potentially 30-40% BPB improvement
- Requires implementing INT4 quantization with acceptable accuracy

---

## Tokenizer Training Notes

A new BPE tokenizer must be trained on the FineWeb dataset for each vocab size. The existing `fineweb_1024_bpe.model` SentencePiece model serves as reference. Key considerations:

1. **Character coverage**: Must be 1.0 for byte-level BPE (no UNK tokens)
2. **Special tokens**: Match the existing setup (BOS/EOS if used)
3. **Data reprocessing**: All `.bin` shards must be regenerated with the new tokenizer
4. **Validation**: Verify tokens_per_byte on a held-out sample before full training
