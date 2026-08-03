# Search Map

## Current map (v4 — 2026-07-30, post-EXP-048/049)

Legend: `✓` useful/settled · `✗` eliminated at current implementation · `~`
measured but inconclusive/conditional · `?` untested.

```
Evidence and validation
 ├── Durable verified-LB ledger + CSV hashes                 ✓ LEADERBOARD.md
 ├── Public split/count granularity (201; 0.4975%/clip)     ✓
 ├── Micro subject-CV for large moves                       ✓
 ├── Fine-grained CV→LB projection                          ✗ block12 broke smooth-offset claim
 ├── Exact marginal controls / paired adoption gate         ✓ process adopted
 ├── All-18-user deterministic outer protocol              ✓ protocol; model reruns pending
 └── Outer-once epoch/model selection                       ? required for new runs

Working representations
 ├── TCN joint+velocity+bone backbone                       ✓ 54-55%
 ├── ST-GCN architecture block                              ✓ largest verified ladder step
 ├── MultiTCN separate joint/motion/bone branches           ✓ 56.60; +1 LB clip
 ├── Adaptive multi-partition ST-GCN                         ✓ 59.56; +3 LB clips
 ├── Invariant IMU                                          ✓ weak, cheap diversity
 ├── Quaternion world-frame IMU                              ✓ 32.07 solo; +0.48 nested
 ├── CTR-GCN topology refinement                             ✗ fold0/fold2 behind Adaptive
 ├── Dual-frame station+canonical skeleton                   ✗ fold0 −1.63 vs Adaptive
 ├── Correct dynamic truncation                             ~ +0.59 solo; +0.04 in block
 ├── Depth/IR ROI CNNs                                      ✗ current supervised recipes
 ├── Upper-body/object crop                                 ✗ zero on target classes
 ├── Dual IR motion-map CNN                                 ~ 44.36/37.67 solo; fusion signs split
 ├── Cross-user supervised contrastive representation       ✗ fold0 −2.82
 ├── Alternative competition-data pretext/teacher           ? new premise required
 ├── Identity screen                                        ✗ fold0 −3.12
 ├── Radar early-cohort view                                ?
 ├── High-res IR/Depth/Thermal MIL branch                   ✗ EXP-050b object .219 vs .32
 ├── Confusion-cluster specialists                          ✗✗ EXP-054 paired 4-seed −0.0096 (−1.27σ), 3/4 neg — CONFIRMED
 ├── Pair-only routing (MAX_CLUSTER=2, tens of clips)       ? different hypothesis, not a retest
 ├── Person selection `first` vs `motion`                   ✗ EXP-057 +0.57±1.71 (0.34σ), 2/4 seeds neg — Q-86 retired
 ├── Pair-only route w/ visual embedding as INPUT           ?
 └── Mechanism behind the rank-1 outlier (181/201)          ? highest-value untested cell

Rejected DG/temporal implementations
 ├── Aggressive geometric skeleton augmentation            ✗
 ├── Pad+mask TCN at T=64                                   ✗ flat at 55.34
 ├── Subject-adversarial MultiTCN (DANN)                    ✗ −3 to −4 on fold 0
 ├── Mixup α=0.2                                            ~ +0.15, split signs; rejected
 ├── BiGRU                                                  ✗ 53.34
 └── Blanket temporal-jitter TTA                            ✗/conditional; per-member gate only

Fusion and inference
 ├── Scalar late fusion                                     ✓ operating point
 ├── Learned logit gate                                     ✗ −6.8 nested
 ├── Fusion transformer / masked SSL transformer            ✗ final gate
 ├── Same-family seed/width accumulation                    ✗ ceiling-scale lever
 ├── Architecture-diverse compact selection                 ~ optional efficiency; cap solved
 ├── Global uniform prior / Sinkhorn                        ✗ −12/−13 public clips
 ├── Recording-group Hungarian                              ✗ at current base
 ├── Ordered-recording transition Viterbi                   ✓ public 109→112 clips
 ├── Repeated-recording probability consensus               ✗ marginal public 112→111
 ├── Exact recording-template forcing/hybrid                ~ fixed +0.85; nested −0.56 vs fallback
 ├── Naive pseudo-label self-training / TENT-GN             ✗
 └── Class-specific low-parameter correction                ?

Deployment
 ├── Exact organizer ruling preserved                       ✗ reply/screenshot missing
 ├── Strict no-pretrained compliance                        ~ relayed, exact reply missing
 ├── Full `a20` serialized artifact ≤100 MB                 ✓ int8 82.696 MB
 ├── Full world25 serialized artifact ≤100 MB               ✓ int8 85.218 MB
 ├── Persistent fp32 model params ≤100 MB                  ✓ streamed max member 10.135 MB
 ├── Quantized test-argmax parity                           ~ a20 1/405; world25 3/405
 ├── Nested compact member subset                           ~ optional efficiency work
 ├── Validated full-data checkpoint strategy               ?
 ├── Distillation                                           ?
 ├── Quantization with payload audit                        ✓
 └── One-command reproducible package                       ~ package+infer exist; clean rerun open
```

### Current frontier (updated 2026-07-31 after EXP-050b/051)

**Estimated search space remaining: ~25%.** Two predeclared screens closed the
largest open branches. What remains is concentrated, not broad.

0. **The model-side branch over skeleton-derived features is closed** (B-021).
   EXP-050b added new visual features and failed; EXP-051 narrowed the decision
   boundary over existing features and failed. Do not open a third variant of
   either without a new premise.
0b. **Highest-value untested cell: the rank-1 mechanism.** The verified board
   shows 181/201 standing 20 clips clear of rank 2, while ranks 2--20 form a
   smooth 161→131 ladder. A detached point usually indicates a different
   method, not a better-tuned one. DA-002's mechanism posteriors were never
   tested and the crazy tier remains starved.

1. Preserve the verified transition champion:
   `0.55721 = 112/201`, derived from the 85.218 MB package.
2. Close repeat consensus after its 0.55223 result and do not spend more
   submissions on template/distinctness variants.
3. Build the all-user high-resolution IR/depth/Thermal object reset and use it
   only as candidate-conditioned evidence.
4. Preserve the streamed bit-exact loader and obtain exact organizer wording
   for transient-dequantization accounting.
5. Obtain the actual Top-15 score, preserve the organizer reply, and reconcile
   the unsupported block10 claim.

---

## Historical map (v1 — 2026-07-27; superseded)

<!--
Legend: ✓ tested (see LOG) · ✗ tested and eliminated · ? untested · ~ partially explored
-->

**Historical estimate: 74% remaining** (recorded before the later experiment and
leaderboard waves; not a current estimate)

```
Data understanding
 ├── Train profiling                                      ✓ (EXP-000b)
 ├── Test profiling (modality availability)               ✓ (EXP-000b: Depth/IR/Skel/IMU ≥404/405; radar 51% empty; thermal -10)
 ├── Test session/timeline structure                      ✓ (EXP-000c: 144 recordings, user-pure, class-DISTINCT within group)
 ├── Test cohort fingerprinting                           ✓ (EXP-000c: 2-way provable E199/L206; 4-way ≈99/100/105/101 moderate)
 ├── Adjacency label prior                                ✗ (EXP-000c: 0/405 same-class neighbors — DEAD)
 ├── Station prior via background matching                ✗ (EXP-000c: recovery ~50% → 0.09 bits — DEAD)
 ├── Depth JET scale (fixed vs per-frame)                 ✓ (EXP-000c: FIXED absolute mapping; scalar depth is metric)
 ├── Label sanity / duplicate detection                   ?
 └── Clip-duration-as-feature                             ~ (7× class spread; test trimmed ~20%, hard-capped 10 s — recalibrate)
Data / preprocessing
 ├── Preprocessing cache (npz per sample)                 ? (Q-00 — required first)
 ├── Skeleton: re-normalization (torso scale, seq-level)  ? (per-frame pelvis-center+floor-align already baked in!)
 ├── Skeleton: multi-person disambiguation policy         ? (pick-first vs largest vs temporal-consistent)
 ├── Skeleton: multi-person-rate as feature               ? (84.6% Comb_hair — discriminative)
 ├── IMU: device pivot + resample + dual-header parsing   ?
 ├── IMU: orientation-invariant features                  ?
 ├── Visual: depth inversion to scalar                    ✓ (in cache; Q-63 proved absolute scale)
 ├── Visual: person-ROI crop via depth foreground         ? (Q-70 — TOP visual lever after EXP-006)
 ├── Visual: resolution / frame sampling / epochs scaling ? (Q-71)
 ├── Thermal: alignment + ironbow decoding                ? (low priority — misfit modality)
 └── Missing-modality handling                            ? (benign in test for big-4)
Augmentation
 ├── Skeleton: bundled 5-aug suite                        ✗ (EXP-001b: −1.6% vs no-aug — component ablation needed)
 ├── Skeleton: per-component aug ablation                 ? (Q-11b/c — mild magnitudes)
 ├── IMU: jitter/scale/rotation/time-warp/channel-drop    ?
 ├── Visual: crop/flip/erase/temporal-crop/RandAugment    ?
 ├── Mixup / CutMix / SDMix                               ?
 └── Modality dropout (fusion robustness)                 ?
Architecture (per modality)
 ├── Skeleton: GRU/TCN                                    ?
 ├── Skeleton: ST-GCN / CTR-GCN-lite (1.5M params)        ?
 ├── Skeleton: multi-stream (joint/bone/motion)           ?
 ├── IMU: 1D-CNN / DeepConvLSTM / transformer             ?
 ├── Visual: 2D-CNN + temporal pool (TSN-style)           ?
 ├── Visual: R(2+1)D / X3D-XS-style factorized 3D         ?
 ├── Visual: dynamic images / depth motion maps           ?
 ├── Radar: any model stream                              ✗ (EXP-000b: 51% empty by cohort, 6.6 pts/frame — B-008)
 ├── Radar: summary stats as aux features                 ? (low priority)
 └── Capacity sweep within 100 MB single-file             ?
Fusion
 ├── Late (logit avg / weighted)                          ?
 ├── Learned gating / attention                           ?
 ├── Mid-level feature concat                             ?
 └── Per-class fusion weights                             ?
Loss
 ├── CE + label smoothing                                 ?
 ├── Class-balanced sampling/loss (12-319 imbalance!)     ?
 ├── Subject-adversarial (DANN)                           ?
 └── Supervised contrastive (paper used it for LOSO)      ?
Training
 ├── Optimizer / LR schedule                              ?
 ├── EMA / SWA                                            ?
 ├── InstanceNorm/GroupNorm vs BatchNorm (LOSO-proven)    ?
 └── Regularization sweep                                 ?
Validation
 ├── Subject-grouped CV handling ragged coverage          ? (Q-01 — foundation)
 └── CV↔LB correlation                                    ? (needs 2+ submissions)
Inference
 ├── Temporal multi-crop TTA                              ?
 ├── Fold/seed ensembling (within single .pth)            ?
 ├── Recording-group Hungarian distinctness assignment    ? (Q-68 — THE metadata lever; simulate on train CV first)
 ├── Per-cohort BN/stat adaptation (2-way E/L split)      ? (Q-54)
 ├── Test-time BN adaptation (TENT)                       ?
 └── Pseudo-label self-training on test                   ? (not in forbidden list; reproduction-risk)
Rules/meta
 ├── ImageNet-init legality ruling                        ? (Q-65 email draft)
 ├── Kaggle submission limit / test balance check         ? (Q-62b)
 └── Efficiency score optimization (10% of final)         ? (later: distill/quantize)
Unknown ideas (literature)
 ├── CUHK-X paper recon                                   ✓ (EXP-000b)
 ├── Compact skeleton/IMU/visual HAR prior art            ✓ (EXP-000b: CTR-GCN, DeepConvLSTM, TSN, Um-aug)
 └── Missing-modality + DG literature                     ✓ (EXP-000b: modality dropout, ShaSpec, SDMix, DANN)
```

### Historical frontier
1. SUBMIT the pair (Atharv, manual) — calibrates CV↔LB and the Hungarian lever with one Kaggle day.
2. Q-70 person-ROI depth pipeline — the visual recipe fix with the largest expected step (32%→? in-domain).
3. Skeleton DG block: Q-11b aug ablation, Q-10 seq normalization, Q-34 mixup — attacks the measured 20-pt gap.
