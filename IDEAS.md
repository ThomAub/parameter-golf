# Parameter Golf: Ideas

## Quantization & Compression

### Tier 1 — Implement first (highest ROI)
- **INT4 Group Quantization (g=128, asymmetric)**: Double effective param budget from ~19M→~38M. Per-group (128 weights/group) with asymmetric scales (learned zero-point) is the sweet spot. Recovers 2-4% over symmetric at int4. Per-input-channel grouping isolates activation outliers for another 1-3%. ~100-150 lines to implement.
- **STE-Based QAT (2-3 epochs fine-tuning)**: Insert fake quantization nodes in forward pass, use straight-through estimator for gradients. Master weights in BF16, quantized for compute only. Recovers 5-15% of int4 PTQ accuracy gap at zero size cost. Key practices: LR 10-100x lower than base, quantize only Linear layers (keep LayerNorm/embeddings float). ~150-200 lines. References: PyTorch `torch.ao.quantization`, bitsandbytes.
- **Mixed-Precision INT4/INT8**: MLP weights → INT4 (bulk of params, tolerant). Attention QKV/O → INT4 or INT8 (sensitivity-dependent). Embeddings → INT8 (high sensitivity). Control tensors (scales, gains) → FP16.

### Tier 2 — If time permits
- **GPTQ (Hessian-aware rounding)**: Sequential layer-by-layer quantization using second-order (Hessian) info from 128-1000 calibration samples. Recovers ~0.5-1% over naive RTN at same bit depth. Post-training only, no training loop changes. ~300-400 lines. Ref: [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq), [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel).
- **Quantization-Aware Distillation (QAD)**: Train quantized student to match full-precision teacher's soft outputs (KL divergence). Loss = αL_task + (1-α)L_distill. Recovers 90-98% of accuracy gap (NVIDIA NVFP4 report). ~1.5-2x training wall clock. ~300-500 lines.
- **LSQ / LSQ+ (Learned Step Sizes)**: Jointly learn quantization scales via gradient descent. Closes PTQ-QAT gap by ~60-70% at 4 bits. Most useful for extreme int2-int3 (MoE expert weights). ~200-400 lines.

### Tier 3 — Speculative / marginal
- **AQLM (Additive Quantization)**: Learnable codebooks, sub-4-bit Pareto-optimal. High complexity (800-1500 lines), slow. Our int4+zlib budget already gives similar compression. Skip unless targeting <2 bits.
- **BitNet 1-bit**: Native ternary {-1,0,+1} training from scratch. Competitive at 2B+ params but not applicable at 50M scale. Int4 budget is generous by comparison.
- **NF4 (NormalFloat4)**: Information-theoretically optimal for normally-distributed weights. Used in QLoRA. Marginal gain over standard int4 for our setting.

### Concrete size budget math
```
50M params × int4 (0.5 bytes/param) = ~25MB raw
  + scales: 50M/128 × 2 bytes (fp16) = ~0.78MB
  → ~25.8MB pre-compression
  + zlib level 9 → ~5-7MB (entropy-coded, ~3-4x ratio on int4)
  + code ~200KB
  = ~5.5-7.5MB total artifact (well under 16MB)
  → Headroom for 80-100M params if needed
```

## Pruning & Sparsity

### Best bets for 16MB / 10-min constraint
- **Wanda (post-training, ~2 min)**: Activation-weighted magnitude pruning. Single forward pass calibration, no retraining. 70% sparsity achievable. Outperforms magnitude pruning significantly (LLaMA-7B: 7.26 vs 17.29 perplexity). Sparse weights compress 2-3x better with zlib. Simplest to implement. Ref: [locuslab/wanda](https://github.com/locuslab/wanda).
- **N:M 2:4 Sparsity (H100 native)**: 50% structured sparsity with native Tensor Core acceleration (1.3-1.8x end-to-end speedup). Well-studied: ~1-2% perplexity increase. Order critical: **sparsify first, then quantize**. Pairs excellently with int4. Ref: NVIDIA TensorRT sparse docs.
- **Gradual Magnitude Pruning (GMP, in-training)**: For 20K steps: stabilize ~500 steps → prune ~10K steps → fine-tune ~9.5K steps. Cosine annealing pruning rate. Achievable: 50-60% sparsity, minimal degradation. Avoid pruning early layers aggressively. ~Low complexity, integrated into training loop.
- **Block Sparsity (4x4 / 8x8)**: Contiguous zero blocks compress 50-80% better under zlib than random unstructured sparsity. 70% block-sparse → ~2.5-3x compression (vs ~2x unstructured). Slightly higher quality cost than unstructured.

### Viable but lower priority
- **SparseGPT (one-shot, Hessian-based)**: 60% unstructured sparsity with <5% perplexity increase. Generalizes to 2:4 and 4:8 patterns. Higher quality than Wanda but slower. Ref: [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt).
- **Structured Dropout (LayerDrop)**: Randomly skip layers/heads during training → network learns importance ordering → prune unimportant permanently at export. 30-40% pruning with <2% quality loss. Elegant but limited scope.
- **Structured Pruning (heads/neurons)**: Remove entire attention heads or MLP neurons via magnitude/Taylor/Fisher. 30% pruning ratio with minimal quality loss. Use Torch-Pruning (DepGraph) for dependency handling.

### Rejected for our setting
- **Lottery Ticket (IMP)**: Needs multiple train-prune-reset cycles. Not practical in 10-min budget.
- **RigL / Dynamic Sparsity**: 15-25% training overhead, marginal gains over static pruning in short runs.
- **Movement Pruning**: Better for fine-tuning, not from-scratch training.

### Recommended pipeline: Sparse + Quantize
```
Order: PRUNE FIRST → then QUANTIZE (critical — reverse order degrades quality)

Option A (post-training, fast):
  1. Wanda 70% sparsity (2 min, single forward pass)
  2. GPTQ int4 quantization (1 min)
  3. Optional: GMP fine-tune 7 min for recovery
  → 8-12x compression, <15% perplexity loss

Option B (hardware-accelerated):
  1. SparseGPT/Wanda to 50% 2:4 sparsity (3 min)
  2. GPTQ int4 quantization (2 min)
  3. GMP fine-tune with 2:4 mask frozen (5 min)
  → H100 Tensor Core acceleration + 16MB size

Option C (training-integrated, best quality):
  1. GMP during training: 50-60% sparsity (built into 20K step schedule)
  2. STE-based int4 QAT last 2-3 epochs
  3. Block-sparse pattern for zlib-friendly compression
  → Best quality, but more complex implementation
```

## Vocabulary & Tokenization
- **Optimal Vocab Size (2048-4096)**: Increase vocab from 1024 to reduce tokens-per-byte by 15-30%, directly lowering BPB. Sweet spot balances embedding table cost against sequence compression (see `research/optimal_vocab_size.md`).
- **Variable-Dimension Embeddings (Matryoshka)**: Assign higher-dimensional embeddings to frequent tokens and lower-dimensional to rare ones, projecting up to model_dim. Exploits the Zipfian distribution of language — top 200 tokens cover ~80% of text.
- **Byte-Fallback Tokenization**: Use a smaller core vocab with byte-level fallback for OOV, avoiding wasted embedding rows. Keeps the embedding table lean while maintaining full coverage.

## Architecture
- **Depth Recurrence (Universal Transformer)**: Reuse the same block weights across multiple layers, paying parameter cost once but getting depth for free. A 9-layer model with 3x weight sharing acts like 27 layers at 9-layer parameter cost.
- **Mixture of Experts (MoE)**: Replace dense MLP with 4-8 small experts and a learned router, activating only 1-2 per token. Zero-cost variant splits existing MLP into 4 experts (same params, half the compute); expanded variant with int4 targets ~35M total params for ~1.05-1.10 BPB (see `research/moe_analysis.md`).
- **Paired-Head Attention on Steroids**: Go beyond GQA — pair query heads that share not just KV but also learned relative mixing coefficients. Cuts KV projection parameters further while maintaining representational diversity through learned head interactions.
- **Manifold Ultra-Connections**: Replace linear skip connections with learned low-rank nonlinear transforms between encoder and decoder halves. Richer information flow across the U-Net skip topology at negligible parameter cost (~rank 16-32 bottleneck).
- **Hyper-Efficient Attention (Linear / Performer)**: Replace softmax attention with a linear approximation for some layers. Frees up FLOP budget to run more training steps within the 10-minute wall clock.
- **Sub-Quadratic Feedforward**: Replace relu^2 MLP with a sparse lookup or product-key memory. Same expressivity at lower parameter count — each token retrieves a small subset of a large implicit weight matrix.

## MoE-Specific
- **ScatterMoE / Megablocks Framework**: ScatterMoE (~700 lines of Triton) is best for single-GPU: no distributed overhead, 2-3x faster than naive PyTorch. Megablocks has faster raw kernels but heavier dependencies (see `research/moe_analysis.md`).
- **Expert-Aware Quantization**: Calibrate each expert's int4 ranges only on tokens routed to it, not the full dataset. Paired with keeping routers in fp16, this is the MoE-specific quantization strategy from QMoE/MC-MoE that preserves routing fidelity.
- **Zero-Cost MoE (Split Existing MLP)**: Split each 512→1024→512 MLP into 8 experts of 512→128→512 with top-2 routing — same total params, half the per-token compute. Fine-grained experts give C(8,2)=28 routing combos vs C(4,2)=6, plus add 1 shared always-on expert (DeepSeek-style).
- **Loss-Free Load Balancing (DeepSeek 2024)**: Dynamic per-expert bias on routing scores instead of auxiliary loss. Eliminates α-tuning and achieves better performance AND better balance than standard auxiliary loss methods.
- **Shared Expert Codebooks**: If experts start from the same initialization, share int4 quantization centroids across all experts. Saves ~30% on scale metadata overhead at no quality cost.

## Training & Optimization
- **FP8 Training (H100) [IMPLEMENTED — Phase 1]**: Use FP8 matmuls for forward/backward passes to nearly double throughput. More training steps in 10 minutes = lower loss at the same parameter count. Enable with `USE_FP8=1`. Uses dynamic per-tensor quantization to `float8_e4m3fn` via `torch._scaled_mm` on SM90+ GPUs. Only applied to large weight matrices (>4096 elements) to avoid overhead on small tensors.
- **Triton Fused Kernels [IMPLEMENTED — Phase 1]**: Enable with `USE_TRITON_KERNELS=1`. Two custom kernels:
  - **Fused ReLU^2**: Merges relu + square into a single Triton kernel, saving one memory round-trip in the MLP forward pass (~10-15% MLP speedup).
  - **Fused RMSNorm**: Triton RMSNorm replaces `F.rms_norm` with a single-pass fused kernel, eliminating intermediate allocations (~5-10% norm speedup).
- **Progressive Growing**: Start training with fewer layers and smaller sequences, then grow. Faster early iterations let the model see more data in the same wall clock budget.
- **Aggressive Learning Rate Schedules**: Use WSD (warmup-stable-decay) with a much shorter stable phase. Matched to the 20K iteration budget, this can squeeze out a few percent lower loss.
- **Distillation from a Larger Run**: Train a 40M+ parameter teacher unconstrained, then distill into the submission model. The student can learn softer targets that compress better than raw data.

## Radical / Speculative
- **Sparse Circuit Discovery & Compression During Training**: Identify and freeze critical computational circuits mid-training, then prune everything else. Combines lottery ticket hypothesis with online structure discovery — train once, compress by finding the winning subnetwork automatically.
- **Decision Tree Distillation**: Distill the final language model into a hybrid architecture mixing neural layers with learned decision trees for frequent patterns. Tree components compress to near-zero size and handle the long tail of predictable n-gram patterns perfectly.
- **Neural Architecture Search (NAS) within Budget**: Use a supernetwork with weight sharing to search over depth, width, head count, and MLP ratio jointly. Finds the Pareto-optimal architecture for the 16MB constraint rather than guessing.
- **Kolmogorov Complexity-Aware Training**: Add a regularizer that penalizes weight entropy directly, encouraging maximally compressible weight distributions. Trains the model to be good AND small simultaneously rather than training then compressing.
- **Tensor Train / Low-Rank Factorization**: Decompose all weight matrices into tensor-train format with learned ranks. Can achieve 3-5x compression on MLP weights with minimal accuracy loss if ranks are tuned per layer.
- **Activation Checkpointing + Wider Model**: Trade compute for memory to train a much wider model within GPU memory, then compress. Wider models compress better than deeper ones due to more redundancy in weight matrices.

## Key Organizer Tips (integrated)
- **16MB is compressed** — more params OK if they compress well. Baseline: ~22M params → <16MB via int8+zlib. Aggressive quant (int6/int5) allows 3x MLP expansion.
- **Custom tokenizer allowed** — smaller vocab (default 1024) saves embedding params. See `data/cached_challenge_fineweb.py`.
- **Sliding window eval** — significant BPB boost at eval time without changing the model. Eval seq len can differ from train seq len.
- **Weight tying across layers** (depth recurrence) — not just embed/head tying. Already in IDEAS above.
- **Muon optimizer** — especially effective for matrix params in constrained settings.

## Investigated & Rejected
- **Encoder / MLM (BERT-style)**: Cannot compute exact BPB — pseudo-log-likelihood is an approximation, and exact marginalization costs O(n) forward passes per sequence (~1000x slower eval). Not viable for this competition format.
- **Discrete Diffusion (MDLM, SEDD, D3PM)**: Provides valid ELBO-based likelihood but loses to autoregressive by 3-10% at small scale. Requires ~1000 denoising steps for tight bounds, making eval 1000x slower. Training convergence is also slower, wasting our 10-minute budget (see `research/encoder_diffusion_analysis.md`).
- **Hybrid Diffusion (PLAID)**: Uses AR model for scoring anyway, so no BPB advantage over pure AR. Only helps generation quality, which isn't scored.
