# Architecture Comparison: Dense vs MoE for Parameter Golf

## Current Baseline Architecture (Dense, 17.1M params → 1.2244 BPB)

```
                        ┌─────────────────────────────────┐
                        │         Token IDs (B, T)        │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │   tok_emb: Embedding(1024, 512) │  524K params (tied w/ lm_head)
                        │         + RMS Norm              │
                        └────────────────┬────────────────┘
                                         │  x0 (saved for resid_mix)
                          ┌──────────────┴──────────────┐
                          │                             │
              ┌───────────▼───────────┐                 │
              │   ENCODER (4 layers)  │                 │
              │   Blocks 0-3          │                 │
              │   Each: Attn + MLP    │─── skips[]      │
              │   w/ resid_mix(x, x0) │    stored       │
              └───────────┬───────────┘                 │
                          │                             │
              ┌───────────▼───────────┐                 │
              │   DECODER (5 layers)  │                 │
              │   Blocks 4-8          │◄── skips[]      │
              │   x += skip_w * skip  │    consumed     │
              │   + resid_mix(x, x0)  │◄────────────────┘
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │      final_norm       │
              │   (RMS Norm, no params)│
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  lm_head = tok_emb.T  │  (tied, 512 × 1024)
              │  + logit_softcap(30)  │
              └───────────┬───────────┘
                          │
                   ┌──────▼──────┐
                   │ CE Loss     │
                   └─────────────┘
```

### Each Transformer Block (×9 unique blocks):

```
    ┌─────────────────────────────────────────────────┐
    │  Block(dim=512, heads=8, kv_heads=4, mlp=2x)   │
    │                                                 │
    │  ┌─ resid_mix ─────────────────────────────┐    │
    │  │ x = mix[0]*x + mix[1]*x0  (2×512 params)│    │
    │  └──────────────────┬──────────────────────┘    │
    │                     │                           │
    │  ┌──────────────────▼──────────────────────┐    │
    │  │ Attention (GQA: 8Q, 4KV)                │    │
    │  │  c_q: 512→512   c_k: 512→256            │    │
    │  │  c_v: 512→256   proj: 512→512           │    │
    │  │  + RoPE(base=10k) + QK RMSNorm          │    │
    │  │  + q_gain (8 scalars)                   │    │
    │  │  Params: 512² + 2×(512×256) + 512²      │    │
    │  │        = 262K + 131K + 131K + 262K       │    │
    │  │        = 786K                            │    │
    │  └──────────────────┬──────────────────────┘    │
    │  x = x + attn_scale * attn_out  (512 params)   │
    │                     │                           │
    │  ┌──────────────────▼──────────────────────┐    │
    │  │ MLP (relu² activation)                  │    │
    │  │  fc:   512 → 1024 (524K)                │    │
    │  │  proj: 1024 → 512 (524K)                │    │
    │  │  Params: 1,048K                         │    │
    │  └──────────────────┬──────────────────────┘    │
    │  x = x + mlp_scale * mlp_out    (512 params)   │
    │                     │                           │
    └─────────────────────┼───────────────────────────┘
                          │
                    ~1.84M params/block × 9 = 16.5M
                    + 524K embed = 17.1M total
```

### Parameter Budget (int8 → 19.4M max, int5/int6 → ~28M max):

```
Component              │ Params   │ int8 bytes │ int5+zstd │ Notes
───────────────────────┼──────────┼────────────┼───────────┤
tok_emb (tied)         │   524K   │    524K    │   ~350K   │ fp16 in PR#180
Attention (×9)         │  7.07M   │   7.07M    │  ~4.7M   │ int6 in PR#180
MLP (×9)               │  9.44M   │   9.44M    │  ~5.0M   │ int5 in PR#180
Scalars/norms (×9)     │   ~23K   │    ~23K    │   ~23K   │ fp32 kept
skip_weights           │    4.5K  │    4.5K    │   4.5K   │ fp32 kept
───────────────────────┼──────────┼────────────┼───────────┤
TOTAL                  │  17.1M   │  ~17.1M    │ ~10.1M   │
+ zlib/zstd overhead   │          │  ~15.8MB   │ ~12.5MB  │
Headroom               │          │   0.2MB    │  3.5MB   │ ← room for more params!
```

---

## Proposed Architecture A: Enhanced Dense (PR#180-style, ~25M params)

```
                        ┌─────────────────────────────────┐
                        │         Token IDs (B, T)        │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │   tok_emb: Embedding(1024, 512) │  524K params (tied)
                        │         + RMS Norm              │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │    SmearGate(512)                │  512 params
                        │    (1-g)*x + g*x_prev            │  bigram blending
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │  BigramHash(10240, dim=128→512)  │  1.31M + 65K proj
                        │  x = x + scale * proj(embed(    │
                        │    XOR(36313*t, 27191*t_prev)    │
                        │    % 10239))                     │
                        └────────────────┬────────────────┘
                                         │  x0
                          ┌──────────────┴──────────────┐
                          │                             │
              ┌───────────▼───────────┐                 │
              │   ENCODER (5 layers)  │                 │
              │   Blocks 0-4          │─── skips[]      │
              │   + VRL (v += λ*v0)   │                 │
              └───────────┬───────────┘                 │
                          │                             │
              ┌───────────▼───────────┐                 │
              │   DECODER (5 layers)  │                 │
              │   Blocks 5-9          │◄── skips[]      │
              │   + XSA (last 4 lyrs) │                 │
              │   + VRL               │◄────────────────┘
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  lm_head (tied embed) │
              │  + logit_softcap(30)  │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Sliding Window Eval  │  stride=64, seq_len=4096
              │  + Causal TTT (LoRA)  │  rank-8 on Q,V,lm_head
              └───────────────────────┘

    New components vs baseline:
    ├── SmearGate:    +512 params
    ├── BigramHash:   +1.38M params
    ├── VRL lambdas:  +9 scalars
    ├── XSA:          +0 params (structural change)
    ├── 10th layer:   +1.84M params
    └── Total:        ~20.3M params → fits int5/int6 + zstd

    Training: seq_len=4096, Muon WD=0.04, SWA(24 ckpts, last 40%)
    Eval TTC: sliding window stride=64 + causal TTT
```

### Parameter Budget (Enhanced Dense):

```
Component              │ Params   │ Quant    │ Compressed  │
───────────────────────┼──────────┼──────────┼─────────────┤
tok_emb (tied)         │   524K   │ fp16     │   ~1.0MB    │
BigramHash embed       │  1.31M   │ fp16     │   ~2.6MB    │
BigramHash proj        │   65K    │ int6     │   ~43K      │
Attention (×10)        │  7.86M   │ int6     │  ~5.2MB     │
MLP (×10)              │ 10.49M   │ int5     │  ~5.6MB     │
SmearGate + VRL + etc  │   ~2K    │ fp32     │   ~8K       │
skip_weights           │   5K     │ fp32     │   ~20K      │
───────────────────────┼──────────┼──────────┼─────────────┤
TOTAL                  │ ~20.3M   │ mixed    │ ~14.5MB     │
+ code (~50K)          │          │          │ ~14.6MB     │
Headroom to 16MB       │          │          │  1.4MB      │ ← could add 11th layer
```

---

## Proposed Architecture B: MoE (Zero-Cost + Expanded)

### B1: Zero-Cost MoE (same params as baseline, fewer FLOPs per token)

```
                        ┌─────────────────────────────────┐
                        │         Token IDs (B, T)        │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │   tok_emb: Embedding(1024, 512) │
                        │   + SmearGate + BigramHash      │
                        └────────────────┬────────────────┘
                                         │  x0
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │  Block (×10, dense attn + MoE MLP)                 │
              │                                                    │
              │  ┌─ resid_mix(x, x0) ─────────────────────────┐   │
              │  └─────────────────────┬───────────────────────┘   │
              │                        │                           │
              │  ┌─────────────────────▼───────────────────────┐   │
              │  │ Attention (GQA: 8Q, 4KV) — same as dense    │   │
              │  │ Params: 786K per layer                      │   │
              │  └─────────────────────┬───────────────────────┘   │
              │  x += attn_scale * out │                           │
              │                        │                           │
              │  ┌─────────────────────▼───────────────────────┐   │
              │  │ MoE MLP (4 experts, top-2)                  │   │
              │  │                                             │   │
              │  │  ┌─────────┐                                │   │
              │  │  │ Router  │ gate: 512→4 (2K params)        │   │
              │  │  │ top-2   │ → weights, indices             │   │
              │  │  └────┬────┘                                │   │
              │  │       │ route tokens to experts             │   │
              │  │  ┌────▼────┬────────┬────────┬────────┐     │   │
              │  │  │Expert 0 │Expert 1│Expert 2│Expert 3│     │   │
              │  │  │512→256  │512→256 │512→256 │512→256 │     │   │
              │  │  │relu²    │relu²   │relu²   │relu²   │     │   │
              │  │  │256→512  │256→512 │256→512 │256→512 │     │   │
              │  │  └────┬────┴────┬───┴────┬───┴────┬───┘     │   │
              │  │       │    weighted sum (top-2)    │         │   │
              │  │       └────────────┬──────────────┘         │   │
              │  │  Per-expert: 262K  │  Total: 4×262K = 1.05M │   │
              │  │  Active: 2×262K = 524K (same FLOP as dense) │   │
              │  └─────────────────────┬───────────────────────┘   │
              │  x += mlp_scale * out  │                           │
              └────────────────────────┼───────────────────────────┘
                                       │
              ┌────────────────────────▼───────────────────────┐
              │  lm_head (tied) + softcap                      │
              └────────────────────────────────────────────────┘

    Parameter count (Zero-Cost MoE):
    ├── Attention (×10): 7.86M
    ├── MoE MLP (×10):  10.49M + 20K router = 10.51M
    ├── Embedding:       524K
    ├── BigramHash:      1.38M
    ├── Misc:            ~25K
    └── TOTAL:           ~20.3M  (same as enhanced dense!)

    But: half the MLP FLOPs → faster steps → more training in 10 min
    OR: reinvest FLOPs into wider model / more layers
```

### B2: Expanded MoE (more params via aggressive quantization)

```
              ┌──────────────────────────────────────────────────────┐
              │  Block (×12, dense attn + MoE MLP)                  │
              │                                                     │
              │  dim=576 (wider than baseline 512)                  │
              │                                                     │
              │  ┌─────────────────────────────────────────────┐    │
              │  │ Attention (GQA: 8Q, 4KV, head_dim=72)       │    │
              │  │ c_q: 576→576  c_k: 576→288                 │    │
              │  │ c_v: 576→288  proj: 576→576                 │    │
              │  │ Params: 1.0M per layer                      │    │
              │  └─────────────────────┬───────────────────────┘    │
              │                        │                            │
              │  ┌─────────────────────▼───────────────────────┐    │
              │  │ MoE MLP (8 experts, top-2) + 1 shared       │    │
              │  │                                             │    │
              │  │  ┌──────────┐                               │    │
              │  │  │ Router   │ gate: 576→8 (4.6K params)     │    │
              │  │  │ top-2    │ + loss-free balancing          │    │
              │  │  └────┬─────┘                               │    │
              │  │       │                                     │    │
              │  │  ┌────▼───────────────────────────────┐     │    │
              │  │  │  Shared Expert: 576→288→576 (332K) │     │    │
              │  │  │  (always active for every token)   │     │    │
              │  │  └────────────────┬──────────────────┘     │    │
              │  │       │           │                         │    │
              │  │  ┌────▼────┬──────┴─┬───────┬─── ... ──┐   │    │
              │  │  │Exp 0   │Exp 1   │Exp 2  │   Exp 7  │   │    │
              │  │  │576→144 │576→144 │576→144│   576→144│   │    │
              │  │  │relu²   │relu²   │relu²  │   relu²  │   │    │
              │  │  │144→576 │144→576 │144→576│   144→576│   │    │
              │  │  └───┬────┴───┬────┴───┬───┴─── ┬ ───┘   │    │
              │  │      └── top-2 weighted sum ────┘         │    │
              │  │                    │                       │    │
              │  │  Per routed expert: 166K params            │    │
              │  │  8 routed experts: 1.33M                   │    │
              │  │  + shared: 332K                            │    │
              │  │  Active per token: shared + 2 routed       │    │
              │  │                  = 332K + 332K = 664K      │    │
              │  │  Total MoE MLP: 1.66M per layer            │    │
              │  └─────────────────────┬───────────────────┘    │
              └────────────────────────┼────────────────────────┘
                                       │

    Parameter count (Expanded MoE):
    ├── Attention (×12):  12.0M
    ├── MoE MLP (×12):   19.9M + 55K routers = 20.0M
    ├── Embedding:        590K (576 dim)
    ├── BigramHash:       1.38M
    ├── Misc:             ~30K
    └── TOTAL:            ~34.0M

    Quantization budget:
    ├── Attn (int6):      ~8.0M compressed
    ├── MoE MLP (int5):   ~10.6M compressed
    ├── Embed (fp16):     ~2.0M
    ├── BigramHash(fp16): ~2.6M
    └── Estimated total:  ~14.8MB  ✓ fits 16MB!
```

---

## Head-to-Head Comparison

```
                    │ Baseline    │ Dense+       │ MoE Zero-Cost │ MoE Expanded  │
────────────────────┼─────────────┼──────────────┼───────────────┼───────────────┤
Layers              │ 9           │ 10-11        │ 10            │ 12            │
Width (d_model)     │ 512         │ 512          │ 512           │ 576           │
Total params        │ 17.1M       │ 20.3M        │ 20.3M         │ 34.0M         │
Active params/token │ 17.1M       │ 20.3M        │ ~15.5M        │ ~22.0M        │
MLP compute/token   │ 1.0x        │ 1.0x         │ 0.5x          │ 0.63x         │
ms/step (est.)      │ 43ms        │ 48ms         │ 38ms          │ 55ms          │
Steps in 10min      │ 13,800      │ 12,500       │ 15,800        │ 10,900        │
Quant scheme        │ int8+zlib   │ int5/6+zstd  │ int5/6+zstd   │ int5/6+zstd   │
Artifact size       │ 15.8MB      │ ~14.6MB      │ ~14.6MB       │ ~14.8MB       │
BigramHash          │ No          │ Yes          │ Yes           │ Yes           │
SmearGate           │ No          │ Yes          │ Yes           │ Yes           │
XSA/VRL             │ No          │ Yes          │ Yes           │ Yes           │
SWA                 │ No          │ Yes          │ Yes           │ Yes           │
Sliding window eval │ No          │ stride=64    │ stride=64     │ stride=64     │
TTT (causal)        │ No          │ Yes          │ Yes           │ Yes           │
────────────────────┼─────────────┼──────────────┼───────────────┼───────────────┤
Expected BPB        │ 1.224       │ ~1.10-1.13   │ ~1.08-1.12    │ ~1.03-1.08    │
  w/o TTT           │ 1.224       │ ~1.13-1.15   │ ~1.11-1.14    │ ~1.06-1.10    │
```

---

## Recommendation: Which Architecture?

### Dense+ (Low risk, strong results)

**Pros:**
- Well-proven by PR #180 and #457
- Simpler implementation — no routing, no load balancing
- Faster to iterate on (no MoE framework dependency)
- All components (XSA, VRL, BigramHash, SWA, TTT) are independently validated

**Cons:**
- Fewer total params than MoE at same artifact size
- Every token pays full MLP compute cost

**Best for:** Conservative approach, guaranteed improvement over baseline

### MoE Expanded (High ceiling, more risk)

**Pros:**
- 34M params vs 20M → significantly more model capacity
- Each token still sees ~22M active params (more than dense baseline)
- Expert specialization may capture different text domains efficiently
- At eval, all experts contribute (no routing waste)

**Cons:**
- MoE at <50M params is under-studied territory
- Expert collapse risk with only 10 min training
- Routing overhead may eat into step speed
- More complex implementation (ScatterMoE + custom quant)
- Quantization of expert weights needs careful calibration per expert

**Best for:** Pushing the frontier, willing to iterate on MoE tuning

### Hybrid: Dense first, MoE second

**Recommended approach:**
1. **Week 1**: Implement Dense+ with all PR#180 techniques + causal TTT
2. **Week 2**: Replace MLP with zero-cost MoE (same params, fewer FLOPs)
3. **Week 3**: Expand MoE with int5 quant if zero-cost MoE shows gains

This de-risks MoE by building on a strong dense baseline first.

---

## Our Bet: Dense+ First, Then MoE Graduation

### Phase 1: Enhanced Dense (~30M params) — Target: 1.10–1.13 BPB

```
┌──────────────────────────────────────────────────────────────────┐
│                     ENHANCED DENSE MODEL                        │
│                                                                  │
│  tok_emb(1024, 512) ──► SmearGate ──► BigramHash(10240)         │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────┐         │
│  │  11 Transformer Blocks × 512d                       │         │
│  │  ┌─────────────────────────────────────────────┐    │         │
│  │  │ GQA(8Q, 4KV) + VRL(λ·v_first)              │    │         │
│  │  │ XSA on last 4 layers                        │    │         │
│  │  │ relu² MLP: 512→1024→512                     │    │         │
│  │  │ U-Net skips, resid_mix(x, x0)              │    │         │
│  │  └─────────────────────────────────────────────┘    │         │
│  └─────────────────────────────────────────────────────┘         │
│       │                                                          │
│       ▼                                                          │
│  lm_head (tied) + softcap(30)                                    │
│                                                                  │
│  QUANTIZATION:                                                   │
│  ├── Embedding:  fp16                                            │
│  ├── Attention:  int5 (middle layers) / int6 (first+last)       │
│  ├── MLP:        int4 with QAT (warmdown-QAT)                   │
│  ├── BigramHash: fp16                                            │
│  ├── Pruning:    5-8% unstructured + fine-tune recovery          │
│  └── Compress:   LZMA -9e (or custom ANS)                       │
│                                                                  │
│  TRAINING:                                                       │
│  ├── seq_len=4096, batch=786K tokens                             │
│  ├── Muon WD=0.04, orthogonal init + muP                        │
│  ├── SWA: 24 ckpts from last 40% warmdown                       │
│  └── Warmdown-QAT: enable int4 STE in last 1200 steps           │
│                                                                  │
│  EVAL (test-time compute):                                       │
│  ├── Sliding window: stride=64, seq_len=4096                    │
│  └── Causal TTT: rank-8 LoRA on Q,V,lm_head (score-then-adapt) │
│                                                                  │
│  BUDGET: ~30M params @ ~4.3 eff. bits/param → ~15.5MB           │
└──────────────────────────────────────────────────────────────────┘
```

### Phase 2: MoE Upgrade (~38M params) — Target: 1.03–1.08 BPB

```
┌──────────────────────────────────────────────────────────────────┐
│                       MOE MODEL                                  │
│                                                                  │
│  Same embedding pipeline (tok_emb + SmearGate + BigramHash)      │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────┐         │
│  │  12 Transformer Blocks × 576d                       │         │
│  │  ┌─────────────────────────────────────────────┐    │         │
│  │  │ Dense Attention: GQA(8Q, 4KV) + VRL + XSA   │    │         │
│  │  │                                             │    │         │
│  │  │ MoE MLP:                                    │    │         │
│  │  │   1 shared expert (576→288→576, always on)  │    │         │
│  │  │ + 8 routed experts (576→144→576, top-2)     │    │         │
│  │  │   Router: 576→8, loss-free balancing         │    │         │
│  │  │   Active: shared + 2 routed = ~660K params  │    │         │
│  │  └─────────────────────────────────────────────┘    │         │
│  └─────────────────────────────────────────────────────┘         │
│       │                                                          │
│       ▼                                                          │
│  lm_head (tied) + softcap(30)                                    │
│                                                                  │
│  QUANTIZATION:                                                   │
│  ├── Experts (routed): int4 with QAT                             │
│  ├── Shared expert:    int5                                      │
│  ├── Attention:        int5/int6 (sensitivity-based)             │
│  ├── Router:           fp16 (NEVER quantize!)                    │
│  ├── Pruning:          5% unstructured                           │
│  └── Compress:         LZMA or custom ANS                        │
│                                                                  │
│  BUDGET: ~38M params @ ~3.5 eff. bits/param → ~15.8MB           │
└──────────────────────────────────────────────────────────────────┘
```

### Phase 3 (Speculative): Depth Recurrence + MoE — Target: <1.00 BPB?

```
┌──────────────────────────────────────────────────────────────────┐
│                 DEPTH-RECURRENT MOE                              │
│                                                                  │
│  3 unique transformer blocks (shared weights)                    │
│  × 4 loops during training (12 effective layers)                 │
│  × 8 loops during eval (24 effective layers!)                    │
│                                                                  │
│  Each block: dense attn + MoE MLP (8 experts + 1 shared)        │
│  Per-loop LoRA adapters (tiny, ~10K params each)                 │
│                                                                  │
│  Unique params: 3 blocks × ~2M = 6M + LoRA adapters             │
│  → Massive freed budget for width: d=768 or d=1024              │
│  → 40M+ effective params at eval in <16MB                        │
│                                                                  │
│  Risk: very hard to train, untested at this scale                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Decisions Needed

1. **Dense or MoE?** → Start dense (Phase 1), graduate to MoE (Phase 2)
2. **Vocab size?** → 1024 (current). BigramHash(10240) substitutes for larger unigram vocab
3. **seq_len?** → 4096 (proven by PR#457, "single largest contributor")
4. **Quantization?** → int4(MLP)/int5-6(attn) with warmdown-QAT. Target ~4.0-4.5 eff. bits/param
5. **Compression?** → Try LZMA first, then custom ANS if worth the effort
6. **TTT?** → Per-document causal TTT with rank-8 LoRA (score-then-adapt, NEVER adapt-then-score)
7. **Pruning?** → 5-8% unstructured magnitude, with fine-tune recovery
