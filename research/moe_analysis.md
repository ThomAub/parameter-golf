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

- **Switch Transformer** (Fedus et al. 2021): Showed MoE gains even at modest scale. Key finding: "the properties studied at large scale were consistent at small scale, even with 2, 4, or 8 experts per layer."
- **Krajewski et al. (ICLR 2025)**: Fine-grained MoE scaling laws from 129M to 3.7B params. Concluded: "With optimal settings, MoE models can always outperform traditional Transformers at any computing budget."
- **MobileMoE** (2024): Demonstrated effective MoE for mobile-scale models (~100M params). Used 2-of-4 routing.
- **JetMoE** (2024): 8B total params, 2B active. Trained efficiently on a single machine.
- **OLMoE** (2024): 7B total, 1.3B active. Competitive with dense 7B on downstream tasks.

### Evidence It Might Not

- At **very** small scale (<50M params), the router overhead and load-balancing loss can eat into gains. Krajewski et al. note "overhead from auxiliary losses can overshadow the benefits of conditional computation" at the low end.
- Expert specialization requires seeing enough diverse data — with only 10 minutes of training, experts may not differentiate enough
- The parameter "budget" in our competition is **compressed model size**, not FLOPs. MoE has more total params, which means more bytes to compress
- We are below the smallest models in published MoE scaling studies (~129M). Expect ~10-20% better loss-per-FLOP vs dense, not the dramatic gains seen at billion scale.

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

**Recommendation: 8 experts, top-2 routing** (or 16 fine-grained half-size experts, top-4). The scaling laws show granularity G=1 (standard MoE) is almost never optimal — more, smaller experts consistently reduce loss. 8 experts gives C(8,2)=28 routing combinations vs only C(4,2)=6 with 4 experts.

DeepSeekMoE's fine-grained approach (splitting N=8 into 16 half-size experts, routing top-4) increases combinations from 28 to C(16,4)=1820, dramatically improving expressiveness at the same compute cost.

### Expert Size

Three strategies:
1. **Same total MLP params**: Split the existing 1024-wide MLP into 4 × 256-wide experts. Zero additional params, pure routing benefit.
2. **Fine-grained**: Split into 8 × 128-wide experts with top-2 routing. Same params, same compute, much more routing flexibility.
3. **Expanded total**: Keep 8 × 256-wide experts, quadrupling MLP params. Requires int4 to fit.

**Strategy 2 is the best risk/reward** — no size increase, better routing combinatorics. Strategy 3 is highest ceiling but needs the int4 prerequisite.

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

Standard auxiliary loss with coefficient **α = 0.01** (Switch Transformers) or **α = 0.001** (DeepSeek):
```
L_balance = α × num_experts × Σ(fraction_routed_i × mean_gate_prob_i)
```

At small scale, expert collapse is a bigger risk (one expert can "win" and starve others). Options:
- **Auxiliary loss**: Start with α = 0.01. Monitor expert utilization; increase to 0.02-0.05 if collapsing, decrease to 0.005 if over-balancing hurts quality.
- **Loss-Free Balancing** (DeepSeek, 2024): Applies a dynamic per-expert bias to routing scores instead of an auxiliary loss. Better performance AND better balance — eliminates α tuning entirely. **Preferred approach.**
- **Expert choice** routing (Zhou et al. 2022): Experts choose tokens instead of tokens choosing experts. Guarantees perfect balance.
- **Jitter noise** on router logits during training (Switch Transformer trick).

### Architecture Extras

1. **Add 1 shared expert** (always active, à la DeepSeek). Ensures common patterns are handled reliably without wasting specialized expert capacity. Especially important at small scale.
2. **Dense first 1-2 layers, MoE for the rest.** Provides stable low-level representations before routing decisions begin.
3. **Higher dropout in expert layers** than dense layers — sparse models are more prone to overfitting.

---

## Quantizing MoE Models

### Do MoE Experts Compress Well?

**Generally yes, but precision floor is ~INT4 with few experts:**

- **QMoE** (Frantar & Alistarh 2023): Achieved sub-1-bit (0.8 bits/param) on SwitchTransformer-c2048 (1.6T params, 2048 experts). Key insight: since only a subset of experts activate per token, quantization errors are naturally attenuated. However, on Mixtral-8×7B (only 8 experts), sub-1-bit caused unacceptable degradation. **INT4 is the practical floor for small MoE.**

- **MC-MoE** (Li et al. 2024): Merges similar experts first (reducing expert count), then compresses with low-rank decomposition. Achieved 2.54 bits avg while maintaining quality.

- **QuantMoE-Bench** findings: Attention layers require higher precision than FFNs. Shared experts need more bits than token-conditioned experts (which can go down to 2-bit). Earlier MoE layers need higher precision than later ones.

- **Router weights should NEVER be quantized** — they're tiny (a few thousand params) and extremely sensitive. Quantization errors in gating get amplified by softmax, causing incorrect expert selection (a discrete structural change, not gradual degradation). Keep in fp16/fp32.

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
- **EAQuant** (2025): Dual-objective calibration minimizing both reconstruction error AND routing KL-divergence between full-precision and quantized routing probabilities
- Implementation: run calibration data through the model, record which tokens route where, then quantize each expert using only its tokens

### Mixed-Precision Across Experts

Not all experts are equal. **MxMoE** and **GEMQ** (2025) show that varying bit-width across experts based on sensitivity outperforms uniform quantization:
- Math-reasoning experts have sparse high-magnitude outliers → need higher precision
- Linguistic experts have smoother distributions → tolerate lower precision
- **Linear-block-level granularity** (within each expert) outperforms expert-level uniform allocation
- Implementation: score each expert's sensitivity, allocate 3-bit to robust experts, 5-bit to sensitive ones, targeting 4-bit average

### Shared Quantization Grids

If experts are initialized from the same dense MLP (common for fine-tuning-based MoE), they share similar weight distributions. Use a **shared codebook** across experts:
- Train one set of int4 centroids for all experts
- Each expert stores only indices into the shared codebook
- Saves ~30% on scale/zero-point metadata overhead

### Key Quantization Papers

- [QMoE](https://arxiv.org/abs/2310.16795) — Sub-1-bit for trillion-param MoE
- [QuantMoE-Bench](https://arxiv.org/abs/2406.08155) — Systematic benchmark of MoE quantization
- [EAQuant](https://arxiv.org/html/2506.13329) — Expert-aware dual-objective optimization
- [MxMoE](https://arxiv.org/html/2505.05799v1) — Mixed-precision per expert
- [MC-MoE](https://arxiv.org/abs/2410.06270) — Expert merging + compression
- [MoE-SpeQ](https://arxiv.org/html/2511.14102v1) — Specialized quantization for MoE

---

## Training Frameworks for MoE

### Single-GPU Options (ranked for our use case)

#### 1. ScatterMoE (recommended for single-GPU)
- **By**: Shawn Tan
- **Key feature**: Triton-based scatter/gather ops, ~700 lines of code. No padding, no token dropping.
- **Single GPU**: **Best option.** Minimal dependencies, no distributed training bloat.
- **Speed**: 2-3x faster than naive PyTorch (ParallelLinear fuses expert transforms + reordering)
- **Integration**: Drop-in replacement for MoE layers, works with standard PyTorch training loops
- **Install**: `pip install scattermoe` or copy from [GitHub](https://github.com/shawntan/scattermoe)

#### 2. Megablocks
- **By**: Stanford / Databricks (Trevor Gale et al.)
- **Key feature**: Block-sparse GPU kernels, 40% faster than Tutel
- **Single GPU**: Good, but carries Megatron-LM integration overhead that's unnecessary for us
- **Speed**: 2-5x faster than naive (best raw kernel speed)
- **Install**: `pip install megablocks`
- **Caveat**: Heavier dependency chain. Better if you outgrow ScatterMoE.

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

#### Newer Options (watch list)
- **FusedXpert** (2025): Builds on ScatterMoE with fused routing + grouped GEMM kernels
- **MoEBlaze** (2025): Reports 2-6x speedups over MegaBlocks with 4x memory reduction
- **PyTorch native grouped GEMM**: Triton persistent cache-aware kernels landing in PyTorch core

### Recommendation

**ScatterMoE** for our single-H100 setup — minimal overhead, Triton-based, drop-in. **Megablocks** if we need the absolute fastest kernels and can tolerate the heavier dependency. The training speed gain (~2-5x on MoE layers) translates directly to more training steps in 10 minutes.

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
