# Quantization & Pruning Deep-Dive for Parameter Golf

## Context

The 16MB artifact budget means every bit matters. The relationship is:
```
fewer bits/param → smaller compressed size → room for more parameters → better BPB
```
This feedback loop is the core optimization: aggressive compression enables larger models.

Current SOTA (PR#180): int5(MLP)/int6(attn) + 3% pruning + zstd-22 → 24.7M params in 15.35MB.

---

## 1. Sub-8-Bit Quantization Landscape

| Format | Levels | Bits | Best for | Notes |
|--------|--------|------|----------|-------|
| INT8 | 256 | 8 | Baseline, safe | Current baseline |
| INT6 | 64 | 6 | Attention | Used by PR#180 for attn |
| INT5 | 32 | 5 | MLP | Used by PR#180 for MLP, zstd loves the zero high bits |
| INT4 | 16 | 4 | Aggressive MLP | ~10% more BPB loss, but 2x more params |
| NF4 | 16 | 4 | Pretrained models | Non-uniform levels matched to Gaussian; QAT may negate advantage |
| FP4 (E2M1) | 16 | 4 | Wide dynamic range | Fewer representable values in typical range; INT4 wins for QAT |
| Ternary (1.58b) | 3 | 1.58 | BitNet-style QAT | Requires 2x width to match quality; extreme compression |

**Key insight**: For QAT-trained models (where weight distributions can be shaped during training), uniform INT quantization is competitive with fancy formats like NF4/FP4. The simplicity of INT also means better compressibility.

### Per-Group Quantization
Instead of per-row scales, use groups of 32-128 weights sharing a scale factor. Reduces quantization error significantly at minimal overhead (~2 bytes per group for the scale). This is the standard approach in GPTQ/AWQ.

**Concrete improvement**: Per-group-32 INT4 quantization could match per-row INT5 quality while using fewer bits. The group scale overhead is 2 bytes / 32 weights = 0.5 bits/param, so effective rate = 4.5 bits vs 5 bits for INT5.

---

## 2. Pruning Strategies

### Current: 3% Unstructured Magnitude Pruning
```python
threshold = torch.quantile(param.abs().float().flatten(), 0.03)
mask = param.abs() < threshold
param.masked_fill_(mask, 0.0)
```

### Recommended: 5-10% with Fine-Tuning Recovery
- Increase pruning to 5-10% of weights
- The zeros compress nearly for free under entropy coding
- Fine-tune for 100-200 steps after pruning to recover accuracy
- Or: integrate pruning into the warmdown phase (prune gradually as LR decays)

### Structured Pruning for MoE
If using MoE, the STUN approach (ACL 2025) is relevant:
1. **Structured**: Remove entire redundant experts (expert pruning)
2. **Unstructured**: Apply magnitude pruning within remaining experts
3. This gives both real parameter savings AND better compressibility

### Semi-Structured (2:4) Sparsity
NVIDIA's 2:4 sparsity pattern (2 of every 4 weights are zero) is hardware-accelerated on H100:
- 50% of weights are zero → excellent compression
- H100 sparse tensor cores give ~2x speedup on the sparse matmuls
- Quality loss is ~1-3% with fine-tuning
- **Combined with int8**: effectively ~4 bits/param with hardware acceleration

**Risk**: 50% sparsity on a small model may be too aggressive. Worth testing.

---

## 3. Compression Beyond zstd-22

### Why zstd Isn't Optimal for Weight Data
ZipNN research showed that **LZ-based dictionary matching is wasted on weight data** — quantized weights lack sequential repetition patterns. The LZ matching phase in zstd finds mostly random matches that actually interfere with the entropy coder.

### Better Options

| Method | Compression | Speed | Complexity | Notes |
|--------|-------------|-------|------------|-------|
| zstd-22 | Baseline | Fast | Drop-in | Current approach |
| LZMA/xz -9e | +1-3% | Slow | Drop-in | Simple experiment |
| Custom ANS | +5-10% | Moderate | High | Tuned to weight histogram |
| Huffman | +3-7% | Fast | Medium | Simple, good for discrete weights |
| Bit-plane separation + zstd | +3-5% | Moderate | Medium | Separate high/low bits |

### Bit-Plane Separation
For int5 values stored as int8, the top 3 bits are always 0 or sign-extended. Separating bit planes before compression lets the compressor exploit this structure:
```python
# Instead of compressing raw int8 bytes:
# Separate into high nibble (mostly zeros) and low nibble (useful data)
high = (weights >> 4) & 0x0F  # mostly 0x00 or 0x0F (sign)
low = weights & 0x0F          # the actual quantized values
compressed = zstd.compress(high + low)  # planes concatenated
```

### Custom Entropy Coding
For a 5-bit quantized model with 32 symbol levels, the optimal encoding is:
- Build histogram of all weight values across the model
- Compute Huffman tree or ANS table from the histogram
- Encode using the custom tables
- Store the table (32 entries × 2 bytes = 64 bytes overhead — negligible)

This eliminates zstd's LZ dictionary overhead and achieves near-entropy-optimal compression.

**Estimated savings**: 5-10% smaller than zstd-22, which translates to ~0.8-1.5MB freed → room for 200K-400K more parameters.

---

## 4. Vector Quantization / Codebook Approaches

### When VQ Beats Scalar Quantization
Below ~3 bits/param, vector quantization dramatically outperforms scalar:

| Method | Bits/param | Quality (relative) | Complexity |
|--------|-----------|-------------------|------------|
| INT4 (scalar) | 4.0 | Good | Low |
| AQLM (additive VQ) | 2.0 | Good | High |
| QuIP# (E8 lattice) | 2.0 | Very good | Very high |
| QTIP (trellis-coded) | 2.0 | Best | Extreme |

### Practical Assessment for Parameter Golf
At our current 4-5 effective bits/param, **scalar quantization is sufficient**. VQ becomes worthwhile only if we push below 3 bits/param. The implementation complexity is high (custom CUDA kernels for dequantization, codebook learning integrated into QAT), and the gains at 4+ bits are marginal.

**Exception**: If we adopt depth recurrence (shared weights used N times), the quantization error in shared weights compounds over N iterations. VQ's lower distortion per bit could be valuable here.

---

## 5. QAT vs PTQ

### The Gap Widens Below 6 Bits

| Bit width | PTQ quality loss | QAT quality loss | Gap |
|-----------|-----------------|-------------------|-----|
| INT8 | ~0.1% | ~0.05% | Small |
| INT6 | ~0.5% | ~0.2% | Moderate |
| INT5 | ~1.5% | ~0.5% | Large |
| INT4 | ~5-10% | ~1-2% | **Critical** |
| INT3 | ~20-40% | ~3-8% | **Massive** |

### QAT Implementation for Parameter Golf
The current approach approximates QAT via SWA during warmdown (weights settle into
quantization-friendly regions). For more explicit QAT:

```python
# Straight-through estimator (STE) during training
def quantize_ste(x, clip_range=15):
    scale = x.abs().amax(dim=-1, keepdim=True) / clip_range
    x_q = torch.clamp(torch.round(x / scale), -clip_range-1, clip_range)
    # Forward: quantized. Backward: pass gradient through as if unquantized
    return x + (x_q * scale - x).detach()

# In the forward pass:
class QATLinear(nn.Module):
    def forward(self, x):
        if self.training:
            w = quantize_ste(self.weight, self.clip_range)
        else:
            w = self.weight
        return F.linear(x, w.to(x.dtype))
```

**Warmdown-QAT**: Enable quantization simulation only during the warmdown phase. This avoids slowing convergence in early training while ensuring final weights are quantization-friendly.

---

## 6. Per-Layer Sensitivity Analysis

### Expected Sensitivity Ordering (most to least sensitive)
1. **Embedding layer** — most sensitive, keep fp16
2. **First attention layer** — high sensitivity (processes raw embeddings)
3. **Last attention layer** — high sensitivity (directly feeds lm_head)
4. **Middle attention layers** — moderate sensitivity
5. **MLP layers** — least sensitive, can be quantized most aggressively
6. **BigramHash** — moderate sensitivity (fp16 in current SOTA)

### Proposed Mixed-Precision Scheme

```
Layer Type              │ Current (PR#180)  │ Proposed Aggressive │
────────────────────────┼───────────────────┼─────────────────────┤
tok_emb (tied)          │ fp16              │ fp16                │
BigramHash embed        │ fp16              │ int8 or fp16        │
BigramHash proj         │ int6              │ int5                │
Attn layers 0, 9       │ int6              │ int6 (sensitive)    │
Attn layers 1-8        │ int6              │ int5                │
MLP all layers          │ int5              │ int4 (with QAT)     │
Scalars/norms           │ fp32              │ fp32                │
Router (if MoE)         │ —                 │ fp16 (never quant!) │
```

### How to Measure Sensitivity
```python
# Per-layer sensitivity sweep
for layer_idx in range(num_layers):
    for component in ['attn', 'mlp']:
        # Quantize only this component to target precision
        model_copy = copy.deepcopy(model)
        quantize_single_layer(model_copy, layer_idx, component, bits=4)
        # Measure BPB degradation
        bpb = eval_val(model_copy, ...)
        sensitivity[layer_idx][component] = bpb - baseline_bpb
```

---

## 7. Theoretical Bits-Per-Parameter Analysis

### How Many Params Can We Fit in 16MB?

| Effective bits/param | Max params | Achievability | Notes |
|---------------------|-----------|---------------|-------|
| 5.0 | 25.6M | Current (int5+zstd) | PR#180 achieves ~24.7M |
| 4.5 | 28.4M | Near-term (int4 MLP + int5 attn) | QAT required |
| 4.0 | 32.0M | Achievable (int4 + custom entropy) | Aggressive QAT |
| 3.5 | 36.6M | Challenging (int4 + heavy pruning + VQ) | Research territory |
| 3.0 | 42.7M | Speculative (mixed ternary/int4) | BitNet-style |
| 2.5 | 51.2M | Very speculative (VQ/AQLM) | Needs custom kernels |

**Sweet spot for us**: 4.0-4.5 effective bits/param → 28-32M parameters. Achievable with:
- MLP layers at int4 with QAT
- Attention layers at int5-int6
- Embeddings at fp16
- 5-10% unstructured pruning
- LZMA or custom entropy coding

### The Param-Quality Tradeoff
At our scale, each additional 1M parameters ≈ -0.003 to -0.005 BPB (estimated from
baseline vs 4-hour run and layer ablations). So freeing 5M params via better compression
≈ 0.015-0.025 BPB improvement. This is comparable to TTT gains!

---

## 8. Actionable Recommendations (Prioritized)

### Tier 1: Easy wins (days of effort)
1. **Try LZMA/xz -9e** as drop-in replacement for zstd-22 (1-3% savings)
2. **Increase pruning to 5-8%** with 100 steps of fine-tuning recovery
3. **Per-layer sensitivity sweep** to find optimal bit allocation

### Tier 2: Medium effort (1 week)
4. **INT4 QAT for MLP weights** with STE during warmdown phase
5. **Per-group-32 quantization** instead of per-row (better granularity)
6. **Bit-plane separation** before compression (3-5% savings)

### Tier 3: High effort, high reward (2+ weeks)
7. **Custom ANS entropy coder** tuned to weight histogram (5-10% savings)
8. **2:4 semi-structured sparsity** with H100 sparse tensor cores
9. **Depth recurrence** with shared weights (dramatic param reduction)
10. **BitNet-style ternary MLP** + int6 attention (extreme compression)

### Combined Tier 1+2 Estimate
Starting from PR#180's 24.7M params in 15.35MB:
- LZMA: save ~0.3MB → +750K params
- More pruning: save ~0.2MB → +500K params
- INT4 MLP: save ~1.5MB → +3.75M params
- Per-group quant: save ~0.3MB → +750K params
- **Total: ~30.5M params in <16MB**
- **Estimated BPB gain: 0.015-0.025 from more params alone**

---

## Sources

- [QLoRA / NF4](https://arxiv.org/abs/2305.14314)
- [BitNet b1.58](https://arxiv.org/html/2402.17764v1)
- [ZipNN — Compression for AI models](https://arxiv.org/html/2411.05239v2)
- [AQLM — Additive quantization](https://dl.acm.org/doi/10.5555/3692070.3692558)
- [QuIP# — E8 lattice codebooks](https://arxiv.org/abs/2402.04396)
- [QTIP — Trellis-coded quantization](https://proceedings.neurips.cc/paper_files/paper/2024)
- [MC-MoE — MoE compression](https://proceedings.iclr.cc/paper_files/paper/2025)
- [STUN — Structured-then-unstructured pruning](https://aclanthology.org/2025.acl-long.671.pdf)
- [EfficientQAT](https://aclanthology.org/2025.acl-long.498.pdf)
- [Mix-QViT — Per-layer sensitivity](https://arxiv.org/abs/2501.06357)
- [Deep Compression (Han et al.)](https://arxiv.org/abs/1510.00149)
- [DietGPU — GPU ANS coding](https://github.com/facebookresearch/dietgpu)
- [Saten — Tensor-train decomposition](https://arxiv.org/html/2505.14871)
- [Recursive Transformers with MoL](https://arxiv.org/pdf/2512.12880)
