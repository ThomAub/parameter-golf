# Encoder & Diffusion Models for BPB: Feasibility Analysis

## The Question

Could we get better BPB by training a non-autoregressive model (masked encoder or discrete diffusion) instead of a standard left-to-right GPT?

---

## Encoder / Masked Language Models (BERT-style)

### Can MLMs compute exact BPB?

**No, not tractably.** This is the fundamental blocker.

An MLM models `P(x_i | x_{context})` for masked positions, but BPB requires the joint log-likelihood `log P(x_1, x_2, ..., x_n)`. To get this from an MLM:

1. **Pseudo-log-likelihood (PLL)**: Sum of `log P(x_i | x_{\i})` — one forward pass per token. This is an *approximation*, not a true likelihood. It double-counts dependencies and systematically underestimates entropy. Not valid for BPB scoring.

2. **Exact marginalization**: Requires computing `P(x)` by factoring the joint via chain rule with arbitrary orderings, which needs `O(n)` forward passes for a sequence of length `n`. For a 1024-token sequence, that's 1024 forward passes vs 1 for autoregressive. Evaluation becomes ~1000x slower.

### Verdict: Not viable

Even if the model were somehow better, the evaluation cost makes it impractical. The competition likely evaluates BPB via a single autoregressive forward pass. An MLM would need a completely different eval harness and would be ~1000x slower.

---

## Discrete Diffusion Models (MDLM, SEDD, D3PM)

### How They Work

Discrete diffusion models corrupt text by progressively masking/replacing tokens, then learn to reverse the process. Key recent models:

| Model | Paper | Approach |
|---|---|---|
| **D3PM** | Austin et al. 2021 | Discrete denoising with transition matrices |
| **MDLM** | Sahoo et al. 2024 | Masked diffusion, absorbing state, continuous-time |
| **SEDD** | Lou et al. 2024 | Score entropy discrete diffusion |
| **PLAID** | Gulrajani & Hashimoto 2024 | Parallelized autoregressive with discrete diffusion |

### Can They Compute Exact Log-Likelihood?

**Yes, but with caveats.**

- **MDLM**: Provides a variational lower bound (ELBO) on log-likelihood. The bound is tight with enough diffusion steps. At 1000 steps, the gap is ~0.01-0.05 nats. However, evaluating the ELBO requires ~1000 forward passes (one per diffusion step).

- **SEDD**: Computes a concrete score-based bound. Same multi-pass evaluation requirement.

- In practice, both report **NLL in bits-per-token** that can be converted to BPB. The evaluation is valid but expensive.

### How Do They Compare to Autoregressive at Small Scale?

**They lose. Consistently.**

From MDLM (Sahoo et al. 2024) on text8:
```
Model          Params    BPC (bits per character)
GPT-2 small    124M      1.07
MDLM           124M      1.10 (+0.03)
SEDD           124M      1.10 (+0.03)
```

From SEDD (Lou et al. 2024) on OpenWebText:
```
Autoregressive: Perplexity significantly lower than diffusion at matched size
Diffusion models close the gap only at very large scale (>1B params)
```

Key findings across papers:
- **At small scale (<100M params), autoregressive wins by 3-10%** in perplexity/BPB
- The gap narrows with scale but never fully closes in published results
- Diffusion models excel at **generation quality** (less repetition, better coherence) but not at raw likelihood
- **Training efficiency**: Diffusion models need more training steps to converge, making the 10-minute wall clock even harder

### Why Diffusion Loses on Likelihood

1. **Information-theoretic ceiling**: Autoregressive factorization `P(x) = ∏ P(x_i | x_{<i})` is exact. Diffusion uses a variational bound that is always ≤ true likelihood.
2. **Capacity allocation**: AR models allocate all capacity to the single next-token prediction task. Diffusion must learn to denoise at many noise levels simultaneously.
3. **Training signal density**: Every token in every position gives gradient signal for AR. Diffusion wastes some capacity on easy denoising steps.

---

## PLAID: The Hybrid Approach

PLAID (Gulrajani & Hashimoto 2024) tries to get the best of both worlds:
- Use discrete diffusion for parallel generation
- But score/evaluate with an autoregressive model

This doesn't help us — if we're using an AR model for scoring anyway, we might as well train AR.

---

## What About Non-Autoregressive for Compression Specifically?

There's a theoretical argument that **order-agnostic models could discover better compression schemes** since they're not locked into left-to-right factorization. In practice:

- **XL-Net** (Yang et al. 2019) explored permutation-based training but was not significantly better at perplexity than GPT-2 at matched size
- **Insertion Transformer** models show no likelihood advantage
- The left-to-right factorization is already optimal for chain-rule decomposition

---

## Bottom Line for Parameter Golf

| Approach | Exact BPB? | BPB vs AR | Eval Cost | Training Speed | Verdict |
|---|---|---|---|---|---|
| **Autoregressive (GPT)** | Yes, exact | Baseline | 1 pass | Fast | **Use this** |
| **MLM (BERT-style)** | No (PLL only) | N/A | ~1000 passes | Fast | Not viable |
| **MDLM / SEDD** | ELBO bound | +3-10% worse | ~1000 passes | Slower | Not competitive |
| **Hybrid (PLAID)** | Via AR component | Same as AR | 1 pass (AR part) | Slower | No benefit |

**Recommendation: Stay autoregressive.** The competition's constraints (small model, BPB metric, 10-min training) all favor standard AR transformers. Diffusion models are exciting for generation but lose on likelihood at this scale.

The only scenario where non-AR might help is if the eval metric were generation quality (coherence, diversity) rather than BPB. It isn't.
