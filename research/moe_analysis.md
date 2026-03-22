# Mixture of Experts (MoE) for Parameter Golf

## Why MoE?

MoE replaces the dense MLP in each transformer block with multiple small "expert" MLPs and a learned router. Only top-k experts activate per token, so:

- **Parameter count is high** (all experts stored) → helps compressed model quality
- **Compute per token is low** (only k experts fire) → helps training speed
- Net effect: **more effective capacity per byte of model size**

The key question: does this tradeoff work at ~17-38M total parameters?

---

## MoE at Small Scale: Does It Help?

### Evidence It Works Small

- **Switch Transformer** (Fedus et al. 2021): Showed MoE gains even at modest scale. Key finding: "scaling the number of experts is complementary to scaling model size."
- **MobileMoE** (2024): Demonstrated effective MoE for mobile-scale models (~100M params). Used 2-of-4 routing.
- **JetMoE** (2024): 8B total params, 2B active. Trained efficiently on a single machine.
- **OLMoE** (2024): 7B total, 1.3B active. Competitive with dense 7B on downstream tasks.

### Evidence It Might Not

- At **very** small scale (<50M params), the router overhead and load-balancing loss can eat into gains
- Expert specialization requires seeing enough diverse data — with only 10 minutes of training, experts may not differentiate enough
- The parameter "budget" in our competition is **compressed model size**, not FLOPs. MoE has more total params, which means more bytes to compress

### Critical Insight for Parameter Golf

The scoring is on **compressed model size + BPB**. MoE trades:
```
More total parameters (larger compressed file)  →  BAD
But better loss per active parameter             →  GOOD if gain > size cost
```

If we have 19.4M params budget (int8) and use 4 experts with top-1 routing:
```
Dense MLP:  512 × 1024 × 2 = 1,048,576 params per layer
MoE (4×):   4 × (512 × 512 × 2) = 2,097,152 params per layer (but 512K active)
```

Each MoE layer costs 2x params but only uses 0.5x FLOPs. With 9 layers, MoE MLPs add ~9.4M params. Total model would be ~26M params — **over budget for int8, but fits int4**.

---

## MoE Architecture Recommendations for Our Scale

### Number of Experts

| Experts | Top-k | Active Fraction | Quality | Overhead |
|---|---|---|---|---|
| **4** | 1 | 25% | Moderate gains | Low |
| **4** | 2 | 50% | Better gains | Medium |
| **8** | 1 | 12.5% | Good gains | Medium |
| **8** | 2 | 25% | Best gains | Higher |
| **16** | 1 | 6.25% | Diminishing | High |

**Recommendation: 4 experts, top-2 routing.** At our scale, 4 experts provides meaningful specialization without fragmenting the limited parameter budget too much. Top-2 gives better gradient flow than top-1.

### Expert Size

Two strategies:
1. **Same total MLP params**: Split the existing 1024-wide MLP into 4 × 256-wide experts. Zero additional params, pure routing benefit.
2. **Expanded total**: Keep 4 × 512-wide experts, doubling MLP params. Requires int4 to fit.

**Strategy 1 is safer** — no size increase, just smarter allocation. Strategy 2 is higher ceiling but needs the int4 prerequisite.

### Router Design

```python
# Simple linear router — almost zero parameter cost
class Router(nn.Module):
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)  # 512 × 4 = 2048 params
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, indices
```

Router cost: `dim × num_experts = 512 × 4 = 2,048 params per layer` — negligible.

### Load Balancing

Use auxiliary loss with coefficient **α = 0.01**:
```
L_balance = α × num_experts × Σ(fraction_routed_i × mean_gate_prob_i)
```

At small scale, load imbalance is a bigger risk (one expert can "win" and starve others). Consider:
- **Expert choice** routing (Zhou et al. 2022): experts choose tokens instead of tokens choosing experts. Guarantees perfect balance.
- **Jitter noise** on router logits during training (Switch Transformer trick).

---

## Quantizing MoE Models

### Do MoE Experts Compress Well?

**Mixed findings:**

- **QMoE** (Frantar & Alistarh 2023): Compressed Mixtral 8×7B to under 1 bit per parameter using grouped quantization. Key insight — experts that see fewer tokens have more redundant weights, sometimes compressing *better* than dense layers.

- **MC-MoE** (Li et al. 2024): "Mixture Compressor for MoE" — prunes less-important experts, then quantizes remaining ones more aggressively. Achieved 2.54 bits avg across experts while maintaining quality.

- **Router weights should NOT be quantized** — they're tiny (a few thousand params) and extremely sensitive. Keep in fp16/fp32.

### MoE-Specific Quantization Strategy

```
Component          Precision    Rationale
─────────────────────────────────────────────────
Router gates       fp16         Tiny, routing-critical
Expert MLPs        int4         Largest params, good compression
Attention layers   int8         Moderate sensitivity
Embeddings         int8         Tied, moderate sensitivity
Scalars/norms      fp32         Tiny, training-critical
```

### Expert-Aware Calibration

Standard GPTQ calibrates on all data equally. For MoE:
- Each expert sees only a subset of tokens (those routed to it)
- **Calibrate each expert only on its own routed tokens** — this gives better quantization ranges
- Implementation: run calibration data through the model, record which tokens route where, then quantize each expert using only its tokens

### Shared Quantization Grids

If experts are initialized from the same dense MLP (common for fine-tuning-based MoE), they share similar weight distributions. Use a **shared codebook** across experts:
- Train one set of int4 centroids for all experts
- Each expert stores only indices into the shared codebook
- Saves ~30% on scale/zero-point metadata overhead

---

## Training Frameworks for MoE

### Single-GPU Options (ranked for our use case)

#### 1. Megablocks (recommended)
- **By**: Stanford / Databricks (Trevor Gale et al.)
- **Key feature**: Block-sparse GPU kernels that batch all expert computations efficiently
- **Single GPU**: Excellent. Originally designed for efficient single-node training
- **Speed**: 2-5x faster than naive PyTorch MoE (no padding waste, fused kernels)
- **Integration**: Works with standard PyTorch, drop-in replacement for MLP layers
- **Install**: `pip install megablocks`
- **Caveat**: Requires Triton; H100 support is good

```python
# Megablocks integration sketch
import megablocks
from megablocks.layers.moe import MoE
from megablocks.layers.arguments import Arguments

args = Arguments(
    hidden_size=512,
    ffn_hidden_size=512,  # per-expert
    num_experts=4,
    top_k=2,
    bf16=True,
)
moe_layer = MoE(args)
```

#### 2. ScatterMoE
- **Key feature**: Uses `torch.scatter` operations, no custom CUDA kernels needed
- **Single GPU**: Good. Pure PyTorch, very portable
- **Speed**: ~1.5-2x faster than naive, slower than Megablocks
- **Integration**: Simplest to integrate into existing code
- **Best for**: Quick prototyping, when Megablocks install is problematic

#### 3. Tutel (Microsoft)
- **Key feature**: Adaptive parallelism, optimized all-to-all for multi-GPU
- **Single GPU**: Works but overhead from multi-GPU abstractions
- **Speed**: Comparable to Megablocks on multi-GPU, slightly slower on single-GPU
- **Install**: `pip install tutel`

#### 4. DeepSpeed MoE
- **Key feature**: Part of the DeepSpeed ecosystem, good ZeRO integration
- **Single GPU**: Heavy dependency, designed for distributed
- **Speed**: Good for multi-node, overkill for single-GPU
- **Not recommended** for our 10-minute single-H100 setup

#### 5. Naive PyTorch (fallback)
```python
# Simple but slow — computes all experts, masks unused
outputs = torch.zeros_like(x)
for i, expert in enumerate(self.experts):
    mask = (routing_indices == i)
    if mask.any():
        outputs[mask] += expert(x[mask]) * weights[mask]
```
- Works but 3-5x slower than Megablocks due to sequential expert loops
- Acceptable for 4 experts, painful for 8+

### Recommendation

**Megablocks** for production quality, **ScatterMoE** as fallback. The training speed gain from Megablocks (~2-5x on MoE layers) translates directly to more training steps in 10 minutes — potentially 30-40% more total steps if MLP is the bottleneck.

---

## Concrete Plan for Parameter Golf

### Phase 1: Zero-Cost MoE (no size increase)

Replace each dense MLP (512→1024→512) with 4 experts (512→256→512), top-2 routing:
```
Dense MLP params per layer:  512×1024 + 1024×512 = 1,048,576
MoE params per layer:        4×(512×256 + 256×512) + router = 1,048,576 + 2,048
                             ≈ same total params!
Active params per token:     2×(512×256 + 256×512) = 524,288 (half the compute)
```

**Result**: Same model size, half the MLP compute → use savings for more layers or larger d.

### Phase 2: Expanded MoE (with INT4)

With int4 doubling our budget to ~38M:
```
Config: d=640, L=12, 8 experts top-2, expert_dim=320
MoE MLP per layer: 8×(640×320 + 320×640) + router = 3,276,800
Attn per layer:    ~1,200,000
Total per layer:   ~4,500,000
12 layers:         ~54,000,000
```

Too many params — dial back to 4 experts or smaller expert_dim. Targeting ~35M total.

### Expected Gains

| Config | Params | Active Params | Compressed Size | Expected BPB |
|---|---|---|---|---|
| Baseline dense | 17M | 17M | ~15.8MB | 1.2244 |
| Phase 1 MoE | 17M | ~13M | ~15.8MB | ~1.18 (more layers from FLOP savings) |
| Phase 2 MoE + int4 | 35M | ~20M | ~15MB | ~1.05-1.10 |
