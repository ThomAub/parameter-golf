# Parameter Golf: Experiment Plan

## Goal
Beat the current merged SOTA (PR#180: 1.1428 BPB, no TTT) and approach/beat
the best unmerged submission (PR#457: 1.1839 BPB with TTT, but uses non-causal
protocol — true causal TTT is likely worse).

**Target: < 1.10 BPB** within the 16MB artifact + 10min train + 10min eval constraints.

---

## Phase 0: Reproduce Baseline (prerequisite)
**Goal**: Verify we can train, quantize, compress, and evaluate correctly.

| # | Experiment | What to do | Success criteria |
|---|-----------|-----------|-----------------|
| 0.1 | Run baseline | Train default config, verify 1.2244 BPB | Match published baseline |
| 0.2 | Understand eval | Read eval harness, understand BPB scoring | Can run eval locally |
| 0.3 | Measure budget | Profile: steps/min, GPU util, memory headroom | Know our compute envelope |

**Time estimate**: 1 session. No code changes needed.

---

## Phase 1: Low-Risk High-Impact (proven techniques from PR#180)
**Goal**: Implement the well-proven techniques that got PR#180 to 1.1428.
Each experiment is independent and can be tested in isolation.

| # | Experiment | Expected gain | Params | Risk | Dependencies |
|---|-----------|--------------|--------|------|-------------|
| 1.1 | **seq_len=4096 training** | 0.03-0.06 BPB | — | Low | Need to verify memory fits |
| 1.2 | **Mixed int5/int6 quantization** | 0.01-0.02 BPB | +5-8M freed | Low | Quantization code changes |
| 1.3 | **BigramHash(10240)** | 0.01-0.02 BPB | +1.38M | Low | New embedding module |
| 1.4 | **3-5% magnitude pruning** | 0.005-0.01 BPB | — | Very low | Pre-quant step |
| 1.5 | **Muon optimizer + WD=0.04** | 0.005-0.01 BPB | — | Low | Optimizer swap |
| 1.6 | **SWA (24 checkpoints)** | 0.003-0.008 BPB | — | Very low | Checkpoint averaging |
| 1.7 | **Sliding window eval (stride=64)** | 0.01-0.05 BPB | — | Very low | Eval-only change |

**Protocol**: Test each in isolation against baseline, then combine winners.

### Phase 1 Integration
Combine all Phase 1 winners into a single run. This is our **Dense+ model**.
- **Expected BPB**: ~1.12-1.16 (conservative), ~1.10-1.14 (optimistic)
- **Expected params**: ~24-26M at mixed int5/int6

---

## Phase 2: Architecture Enhancements (from PR#457 + research)
**Goal**: Layer on architectural improvements that require model changes.

| # | Experiment | Expected gain | Params | Risk | Dependencies |
|---|-----------|--------------|--------|------|-------------|
| 2.1 | **XSA (last 4 layers)** | 0.005-0.015 BPB | -tiny | Medium | Attention code mod |
| 2.2 | **VRL (value residual learning)** | 0.005-0.01 BPB | +tiny | Low | Store layer-0 values |
| 2.3 | **SmearGate** | 0.003-0.008 BPB | +512 | Low | Embedding-level module |
| 2.4 | **Orthogonal init + muP** | 0.005-0.01 BPB | — | Low | Init code changes |
| 2.5 | **Wider model (d=576-640)** | 0.01-0.03 BPB | +3-8M | Medium | Needs int4/int5 headroom |

**Protocol**: Test 2.1-2.4 on top of Phase 1 Dense+ model. Test 2.5 only if
param budget allows after quantization gains.

### Phase 2 Integration
- **Expected BPB**: ~1.08-1.12 with all Phase 1+2 techniques, no TTT
- This should match or beat PR#180's 1.1428 with room to spare

---

## Phase 3: Test-Time Training (the big lever)
**Goal**: Add causal TTT during the 10-min eval window.

| # | Experiment | Expected gain | Risk | Dependencies |
|---|-----------|--------------|------|-------------|
| 3.1 | **Causal TTT baseline** | 0.015-0.025 BPB | Medium | Eval harness changes |
| 3.2 | **TTT adapter: rank-8 LoRA** | best quality/speed | Low | LoRA on MLP layers |
| 3.3 | **TTT adapter: full last-layer** | +0.005 BPB? | Medium | More params to adapt |
| 3.4 | **TTT learning rate sweep** | 0.005-0.01 BPB | Low | Hyperparameter search |
| 3.5 | **Cross-document TTT** | 0.01-0.035 BPB | Medium | Process docs in order |

**Critical**: Must use **causal protocol** (score-then-adapt, never adapt-then-score).
PR#457 likely uses non-causal, which inflates their 1.1839 result.

### Phase 3 Integration
- **Expected BPB**: ~1.06-1.10 with Phases 1+2+3
- TTT adds ~10min eval overhead — must fit within budget

---

## Phase 4: Compression & Quantization Frontier
**Goal**: Push the 16MB boundary to fit more parameters.

| # | Experiment | Expected gain | Risk | Dependencies |
|---|-----------|--------------|------|-------------|
| 4.1 | **LZMA/xz-9e compression** | +1-3% space (→ more params) | Very low | Drop-in replacement |
| 4.2 | **INT4 MLP QAT** | +6-10M more params | Medium | QAT training loop |
| 4.3 | **Per-layer quant sensitivity sweep** | 0.005-0.01 BPB | Low | Ablation runs |
| 4.4 | **8-10% pruning + fine-tune** | +1-2% space | Low | Prune → retrain |
| 4.5 | **Bit-plane separation** | +3-5% compression | Medium | Custom packing code |

**Key insight**: Every 1% of compression freed = ~250K more params = ~0.002 BPB.

### Phase 4 Integration
- **Expected param budget**: 28-32M params in 16MB (up from 24.7M)
- Use freed params for: wider model, more layers, or larger vocab

---

## Phase 5: Advanced Architecture (higher risk, higher reward)
**Goal**: Explore techniques that could break through 1.05 BPB.

| # | Experiment | Expected gain | Risk | Dependencies |
|---|-----------|--------------|------|-------------|
| 5.1 | **Depth recurrence (3 blocks × 3 loops)** | 0.03-0.08 BPB | High | Major arch change |
| 5.2 | **Mixture of Depths (MoD)** | 0.01-0.02 BPB (via faster training) | Medium | Per-layer router |
| 5.3 | **Vocab expansion (V=2048-4096)** | 0.02-0.05 BPB | Medium | Tokenizer change |
| 5.4 | **Longer context eval (RoPE NTK)** | 0.005-0.025 BPB | Medium | Extrapolation code |
| 5.5 | **Depth ensemble** | 0.005-0.015 BPB | Low | Needs 5.1 first |

**Note on 5.1 (depth recurrence)**: This is the highest-ceiling technique — reusing
3 blocks across 3× loops gives 27 effective layers at 9-layer param cost. Frees ~11M
params for width (d=768+). But it's a major rewrite and training stability is uncertain.

**Note on 5.2 (MoD)**: Zero extra params. A 512-param router per layer decides which
tokens skip. 30-50% faster training → more steps → lower loss. Very attractive risk/reward.

---

## Phase 6: MoE (only if Phases 1-5 plateau)
**Goal**: Explore MoE as a last resort for squeezing more capacity.

| # | Experiment | Expected gain | Risk | Dependencies |
|---|-----------|--------------|------|-------------|
| 6.1 | **Zero-cost MoE (split MLP, same params)** | 0.01-0.02 BPB | High | MoE routing code |
| 6.2 | **Shared + 4 routed experts, top-1** | 0.01-0.03 BPB | High | Phase 4 quant needed |
| 6.3 | **Expert-aware INT3-4 quantization** | needed to fit | High | Custom quant code |
| 6.4 | **TTT router rewiring** | 0.005-0.01 BPB | Medium | Needs 6.1+Phase 3 |

**Reality check**: MoE at 20-40M is uncharted. Dense models hold all current records
in this competition. MoE is Plan B, not Plan A.

---

## Experiment Execution Order (Critical Path)

```
Week 1: Phase 0 + Phase 1 (reproduce baseline, proven techniques)
         ├── 0.1-0.3: Baseline reproduction
         ├── 1.1: seq_len=4096 (biggest single gain)
         ├── 1.2: Mixed quantization (unlocks param headroom)
         ├── 1.5: Muon optimizer
         ├── 1.3: BigramHash
         ├── 1.4+1.6: Pruning + SWA (quick wins)
         └── 1.7: Sliding window eval
         → Integration run → expect ~1.12-1.16 BPB

Week 2: Phase 2 + Phase 3 (arch enhancements + TTT)
         ├── 2.1-2.4: XSA, VRL, SmearGate, muP (parallel)
         ├── 2.5: Width increase if budget allows
         ├── 3.1-3.2: Causal TTT with LoRA
         └── 3.4-3.5: TTT hyperparams + cross-doc
         → Integration run → expect ~1.06-1.10 BPB

Week 3: Phase 4 + Phase 5 (push boundaries)
         ├── 4.1-4.2: LZMA + INT4 QAT (more params)
         ├── 5.1 or 5.2: Depth recurrence OR Mixture of Depths
         ├── 5.3: Vocab expansion (if tokenizer change is feasible)
         └── 5.4: Longer context eval
         → Integration run → target < 1.05 BPB

Week 4+: Phase 6 (MoE, only if needed)
         └── Only if Phases 1-5 have plateaued
```

---

## Decision Points

### After Phase 1 Integration
- If BPB > 1.18: Something is wrong. Debug before proceeding.
- If BPB 1.14-1.18: On track. Proceed to Phase 2.
- If BPB < 1.14: Ahead of schedule. Consider skipping to Phase 3.

### After Phase 2 Integration
- If BPB > 1.14: Phase 2 techniques not helping. Focus on Phase 3 (TTT) and Phase 4 (compression).
- If BPB 1.08-1.14: Good. TTT should push below 1.10.
- If BPB < 1.08: Excellent. TTT could push below 1.05.

### After Phase 3 Integration
- If causal TTT gain < 0.01: TTT overhead not worth it. Drop TTT, focus on Phase 4-5.
- If causal TTT gain > 0.02: TTT is a keeper. Optimize TTT speed for more iterations.

### MoE Go/No-Go (before Phase 6)
- Only proceed if: (a) dense model is at 1.08+ BPB AND (b) param budget allows 30M+ with INT4.
- Skip MoE if: dense model already < 1.06 BPB (diminishing returns vs. complexity).

---

## Tracking & Measurement

Every experiment must record:
1. **BPB on validation set** (primary metric)
2. **Artifact size** after compression (must be ≤ 16MB)
3. **Training wall clock** (must be ≤ 10 min on 8×H100)
4. **Eval wall clock** (must be ≤ 10 min on 8×H100)
5. **Parameter count** (total and active)
6. **Delta vs. previous best** (ablation)

Results go in `research/experiment_results.md` as a running log.

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| seq_len=4096 OOMs on H100 | Blocks Phase 1 | Gradient checkpointing, reduce batch size |
| INT4 QAT training instability | Blocks Phase 4 | Fall back to INT5 with pruning |
| Causal TTT doesn't help much | Loses ~0.02 BPB | More compute → Phase 4-5 techniques |
| Depth recurrence training collapse | Blocks Phase 5 | Use warmup schedule, fall back to MoD |
| MoE expert collapse at small scale | Blocks Phase 6 | This is why MoE is last resort |
| Compression exceeds 16MB | Submission invalid | Always test artifact size before submitting |

---

## Summary: Expected BPB Trajectory

```
Baseline:              1.2244 BPB
After Phase 1:         ~1.12-1.16 BPB  (proven techniques)
After Phase 2:         ~1.08-1.12 BPB  (arch enhancements)
After Phase 3 (TTT):   ~1.06-1.10 BPB  (test-time training)
After Phase 4 (quant):  ~1.04-1.08 BPB  (more params via compression)
After Phase 5 (adv):    ~1.00-1.05 BPB  (depth recurrence / MoD)
Phase 6 (MoE):         ~0.98-1.03 BPB  (speculative, only if needed)
```
