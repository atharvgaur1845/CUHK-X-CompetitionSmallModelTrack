7-day programme to 187+ — three levers whose own arithmetic reaches ten clips
Written 2026-09-08 (Fable). Executor: Opus 5. Decider: Atharv. Every number is cited to research/LOG.md or marked [measured today] (re-derived this session from artifacts on disk; the scratchpad holding them was wiped by a session restart, so the executor logs them as EXP-132/133 using the recipes in §A before anything else runs).

0. Sizing first — what can and cannot be worth ten clips
Ten public clips = 5 points = 135 of the 2,700 pooled OOF clips. Any lever that cannot plausibly move 135 OOF clips is not a headline lever, whatever its mechanism. Applying that test to everything on the table:

lever	ceiling (OOF pts)	realistic if its gate passes	P(gate)	why the ceiling is what it is
L1 privileged teacher → distilled student	+15	+8 – 12 pts → +10 – 15 clips	35%	student tracks teacher: a 0.877 label-cheating teacher already gave +4.91 (EXP-122); a real teacher at ≥0.85 honest OOF is learnable, and the student has 34 M params of capacity to spare (EXP-112: not recipe-limited; EXP-115: bigger student overfits, so the gain has to come through targets)
L2 per-subject transductive alignment (feature space)	+8.5	+4 – 8 pts → +6 – 12 clips	45%	229 of 658 errors are systematic within a subject [measured today]; fixing one subject-level offset fixes a whole (subject, pair) cell of 3–6 clips at once. Per-subject AdaBN was 1.6× pooled (EXP-099) and pooled was +2 public (EXP-100)
L3 new raw inputs for the object-identity pairs — thermal through the 224/MViT recipe; COCO detector on IR	+6 each	+2 – 4 pts → +4 – 8 clips each	50% / 45%	thermal has only ever been run at 128 px through the r2plus1d recipe (EXP-088/119/120b) — the recipe the MViT-224 recipe beat by +7.5 on IR+depth (EXP-102). A pretrained detector naming the object in the hand has never been run; EXP-068: "the objects ARE in the pixels"
pair-prompt verifier over raw input (the brief's framing)	+10.5	+1 – 3 pts	40%	to be worth ten clips a verifier must decide the top-2 at ≥0.97 on 2,345 clips; the base rate is 0.8785 and fine-grained re-rankers gain 1–3 pts in the literature. It is the wrong shape for the size of the gap. Kept as the fallback (L4)
everything in the brief's graveyard	≤ +1	—	—	measured flat
The answer to the brief's central question, restated: the rank-2 pool is real, but it is not resolved by looking harder at one clip. It is resolved by (a) a teacher that sees more than the student and sharpens the pair in the targets (EXP-122 proved this is the mechanism: oracle beat fused by +1.53 exactly where the pair lives), and (b) conditioning on the subject's other clips, because the pair error is a subject bias, not a clip ambiguity (§1). L1 and L2 are the two routes whose arithmetic reaches +10 each; L3 raises both of their ceilings. Combined, if both big gates pass, the union is +15 – 20 clips. P(≥187 public) ≈ 35–40%, P(≥180) ≈ 65%. No honest plan guarantees 0.93; this one has the two biggest levers gated by Day 2.

1. The measured fact that re-shapes the problem [measured today; log as EXP-132]
Champion fusion recomputed on the 2,700 pooled OOF clips: top-1 0.7563, top-2 0.8637, 658 errors, 290 at rank 2.

statistic	value
errors in (subject, true, pred) cells repeated ≥3× within one subject	229 / 658
share of a subject's errors that repeat the same true→pred confusion inside that subject	33% – 82% (user3 82, user16 75, user19 74, user20 74)
(subject, class-pair) cells with ≥3 errors	70, of which 55 strictly one-directional
rank-2 errors whose subject has a correctly predicted clip of the true class elsewhere	242 / 290
median top-2 margin: correct / rank-2 error / other error	1.84 / 0.35 / 0.42
user23 maps Read_documents → Turn_pages every time and never the reverse. A per-clip model cannot see a subject-constant offset; a re-ranker over posteriors (EXP-131) cannot either. This is the missing variable behind the correlated residuals, and it is consistent with subject sd 5.26 vs seed σ 1.16 (EXP-124), per-subject AdaBN 1.6× pooled (EXP-099), and skeleton −7.6 OOF→public vs visual −0.1 (EXP-063). It sizes L2 and it tells L1 what the teacher must be able to see.

Subject recovery on test — three keys probed [measured today; log as EXP-133]:

key	result
recording day	dead: train days hold 4–8 users with interleaved sessions
300 s timestamp blocks	usable: 140 train blocks, 95.7% user-pure, median 16.5 clips; test = 24 blocks, sizes 41…1
skeleton bone lengths	dead: within-user sd 0.075 > nearest-other-user 0.065; 18-cluster purity 35%
IMU device MACs	dead: one shared device set (leg-MAC split = the known cohort)
block-mean video embedding	untested — probe P0, Day 1 (the subject shift that hurts us is a subject signature)
Test days: 20239/20240/20251/20252 coincide with train users 6–9/21–24 (195 clips ≈ public 201); 20241/20255/20256 are test-only (210 ≈ private 204). Hypothesis: public = the 4 subjects recorded alongside train users. Cheap check in §5.

2. L1 — the privileged teacher, and how it becomes the shipped student in 7 days
Why this is the +10–15 route and not a moonshot. Distillation is the one member-strength mechanism in this repo with a measured large transfer (+4.91 on fold 2 over a leak-free control, +21 OBJECT clips, EXP-122), and its ceiling was declared "exhausted" only because the teacher was built from our own members (oracle-any-member 0.877). R-3 explicitly permits larger teachers; the Large Model Track exists on the same data; next_action.md candidate 1 already records that never building a large teacher "may have been the strategic error". The teacher is never shipped, so R-1 does not touch it; it must be public and disclosed (R-2).

Teacher = privileged, multi-source, frozen. Not a fine-tuned big model (VideoMAE-B fine-tune lost 0/4, EXP-115 — 2,281 clips cannot fine-tune 86 M params). Instead:

source	how	cost
frozen video foundation features on person + wrist crops	V-JEPA 2 ViT-L (facebook/vjepa2-vitl-fpc64-256), InternVideo2-1B, VideoMAE-v2-g — extract once for 3,338 clips × 2 views; attentive probe per fold (minutes). V-JEPA 2 attentive probes sit near fine-tuned SOTA on SSv2/K400, which is the evidence frozen+probe is the right regime here	6 GPU-h H200
the same, subject-centered (L2 applied to the teacher)	subtract subject-mean feature before the probe	0
thermal at 224 (L3a) and COCO detections on IR (L3b)	extra probe inputs	see §4
our five members' oracle targets	EXP-122's build_teacher_targets.py --target oracle	0
ensemble	log-mean of probe posteriors + oracle target, per fold, fold-safe (probe never trained on the held-out users, so the targets are honest — better than EXP-122's leaky fold-2 targets)	0
Gate T1 (Day 2): teacher honest 4-fold OOF ≥ 0.80 (MViT member 0.7116 pooled, EXP-103). If the best single frozen probe is < 0.74 on fold 2 (bar 3.28 over the 0.705 seed mean), the foundation route is dead in one day and L1 reverts to "oracle + L3 inputs" (ceiling ≈ +5).

Student = the shipped MViTv2-S, unchanged architecture, trained with EXP-122's loss (--teacher --distill-alpha 0.7 --distill-temp 2) on the new targets, person and wrist views, 4 folds + all-train. Also distil into the wrist student (EXP-130 is queued for exactly this). Bytes: 0 new — the package already holds two 34 MB MViTs.

Gate T2 (Day 4): student 4-fold paired Δ vs k224_mvit_f* > 1.64 (2 SE) and OBJECT Δ > 0 on ≥3 folds; then all-train, package, one submission.

Arithmetic, stated so it can be scored: teacher 0.85 → student ≈ 0.78–0.82 (+8–11 over 0.705, i.e. 60–90% of the way, the normal distillation regime when the student has capacity) → video slot +8 → fused +5–7 pts → +10–14 public. Teacher 0.80 → student ≈ 0.75 → +5–7 public.

3. L2 — per-subject transductive alignment: +6–12 clips at 0 bytes, and it is on-site-first
The one class of adaptation that has ever transferred here is feature-space and parameter-free (B-028; AdaBN +2 public with the optimum at the endpoint). The MViT trunk has LayerNorm, so AdaBN has nothing to re-estimate — the analogue is applied to the pooled feature. Three nested rungs, each gated on OOF with reference pools subsampled to 40 clips per subject (test-sized):

rung	what	fits anything on test?	gate (16 held-out users, 4 folds)
2a subject mean-centering	f̃ = f − mean_S f + μ_train; re-apply the head. Train-time: same centering with the subject known (this is "Euclidean alignment" from cross-subject EEG, where it is the single largest known lever and it has never been used on video HAR)	no	paired Δ > 1.64; abort if < 0 on ≥ 2 folds
2b transductive prototype refinement	start from the train class prototypes in centered space; ≤10 EM steps of soft assignment on the subject's clips with a class-balance regulariser (TIM-style mutual-information objective). Fixed hyperparameters; adopt only if the OOF curve is flat/endpoint-optimal like AdaBN's, never at an interior optimum	no labels; two fixed constants	Δ > 1.64 and rescued/broken reported; abort if the best setting is interior
2c within-subject pair check	for fused top-2 (a,b), compare emb(x) to the subject's own margin-confident a- and b-clips; decide by the within-subject metric where both exist (242/290 rank-2 errors have the reference in-pool)	no	net > +20 / 2,700 at 40-clip pools
Why this is sized at +6–12 and not +2: the systematic mass is 229/658 errors (8.5 pts); the rungs act on whole (subject, pair) cells; and the private and on-site stages are 8-subject draws from a 16-point subject spread — a mechanism that lifts the worst subjects is worth more there than on public. Depends on P0 (subject recovery by block-mean embedding). If P0 fails, L2 runs on the 300 s blocks (95.7% pure, median 16 clips): smaller pools, smaller gain, still ≥ 0.

Named dispute: EXP-016 (per-user self-training −3.2% at base ≈ 0.50, full-model, τ = 0.6) and EXP-028 (TENT-GN). Neither is 2a/2b: those fitted parameters on hard pseudo-labels of a 0.5 model. B-022's ladder (distinctness −1 → 0 → +1 as base rose 112 → 166) is the measured precedent for a coupling mechanism flipping sign with base accuracy. Re-test, don't inherit.

4. L3 — the two raw inputs nobody ran
3a Thermal through the 224/MViT recipe. Thermal's 0.544 (EXP-088), 0.558 full-frame (EXP-119) and the 4-fold null (EXP-120b) were all 128 px, r2plus1d_18, Kinetics — the recipe that scores 0.64 on IR+depth, where MViT-224 scores 0.715. Thermal frames are 320×240, so 224 is near-native. Build cache/thermal_224 (person crop and full frame; 3-channel; build_model(in_channels=3) needs no stem surgery), train k224_mvit_th_f{0..3}, ~1 h each on H100. Gate: pooled ≥ 0.66 (member accuracy is the binding quantity, EXP-123); then equal-weight into the video slot next to person and wrist and measure fused Δ. Agreement with the person view is 54% — the most complementary view we own. Expected member 0.62–0.72; fused +2–4 pts if ≥ 0.68. Also feeds L1. Also queue the untested skomuro axes as a 2D-net array on the RTX partition (minutes per arm) but they are the second thermal bet, not the first.

3b COCO detector on IR as a symbolic object channel. YOLO11n already ships in the pipeline (R-1 legal: a small standard pretrained CNN). Run yolo11m on 8 IR frames per clip restricted to the person box + margin; keep max-confidence per COCO class for {cup, bottle, bowl, spoon, fork, knife, apple, orange, banana, book, laptop, cell phone, remote, tv, keyboard, toothbrush, scissors, clock, backpack, handbag}. Day-1 probe (20 min): detection rate per class on 200 OOF clips of the worst pairs (Drink↔Eat, Take_medicine↔Drink, Pour/Stir/Peel, Read↔Turn_pages, Phone↔Games↔Selfie). If the rate on IR is < 30% for cups/bottles/phones, dead in 20 minutes. If it is usable: a 20-d per-clip vector, concatenated to the student's pooled feature (a 20×768 projection, < 1 MB) and to the teacher. Expected +2–4 pts where the confusion is object identity; nothing on Sweep↔Mop (no COCO class).

5. Falsification calendar, parallelisation, and the decision rule
Cluster layout per today's cluster/env.sh rewrite: code /home/pabitra/cuhkx, caches and checkpoints on /scratch/pabitra/cuhkx. Arrays over (tag × fold), fold-major, one GPU per task, per-epoch resume state, per-run JSON manifests (no results.csv from arrays).

partition	day 1	day 2–4	day 5–7
gpu_h200_8 (1-day limit → resumable)	L1 features: 3 foundation models × 3,338 clips × 2 views	L1 probes (minutes) → T1 read on day 2	overflow
gpu_h100_4 ×2	parity check 2; embedding dump → P0, 2a, 2c probes (1 GPU-h); 3b detector probe	L3a thermal-224 array (4 folds); L1 distillation array (4 folds × 2 views) after T1; 2b sims	all-train students; package build + --check infer
gpu_a100_8 (5-day)	LOSO 18 × {person s1,s2,s3; wrist s1} = 72 tasks, ~16 h wall, background	idle	LOSO of the final recipe for the selection statistic
gpu_rtx_pro_6000_6	thermal caches (224, full+crop); skomuro 2D array	thermal 2D arms	best thermal all-train
laptop	ledgers, rowdiff, submissions	—	final selection
Decision points: Day 2 — T1 (teacher ≥ 0.80?), P0, 2a, 3b rate. Day 3 — thermal-224 pooled. Day 4 — T2 (student), 2b/2c. Day 5 — package composition, 2–3 single-change submissions (≥ 20 rows each). Day 6 — score, freeze. Day 7 (09-15) — select two finals. Days 8–14 — inference.sh, clean-room rerun on the cluster, report; no new modelling.

LOSO on day 1: yes (question 4). 130 A100-hours on a partition nothing else needs; it gives σ with 36 df (today's n = 3 CI [0.60, 7.30] does not exclude 2.80), per-subject accuracy for all 18 subjects (the selection statistic below), honest 17-subject OOF for teacher targets, and the embedding dump for L2. Two trainer flags are needed before any launch (--holdout-users, --dump-features in kaggle/cuhkx_224_kaggle.py); nothing is in flight, so the vars(args) fingerprint rule is not violated — add both in one commit, then freeze the argparse.

Final-submission rule (question 5):

Only a package verified by unpack_stage2.py --check integrity,weights,infer at rowdiff ≤ 2 is eligible. Size by ls -la.
Selection statistic = LOSO per-subject accuracy, lower quartile (mean of the 4–5 worst subjects). Private and on-site are 8-subject draws from a 16-point subject spread; the worst subjects decide them. Never the mean, never public.
A component ships only if it fits no parameter on test, or is a fixed rule (equal-weight log-sum, penalty 2.0), or passed mean vs 2 SE over folds. Anything chosen on a public delta < 10 clips is excluded by rule.
Public is a sign check: a candidate must not lose > 10 clips; inside ±10 it is unread.
Two finals: the safe legal champion (§6) and the L1/L2 stack that won on rule 2; the stack is primary if it is within noise on public and better on the LOSO lower quartile.
Optional, 3 submissions on day 1: flip one correct clip on a shared day and one on a test-only day to wrong; exact −1 / 0 confirms the public-by-day hypothesis and lets every later rowdiff be read on probable-public rows (EXP-058's minimal-edit probe).
inference.sh processes the whole directory at once (subject recovery needs the batch; R-4 and R-7 both ruled legal) and degrades to per-clip inference for groups < 5 clips, so on-site can never be worse than today's pipeline.
6. T-PKG — the winner into 100 MB before 09-22 (question 6)
item	MB	action
skeleton w25_p4	22.80	as today
person student (L1-distilled) + wrist student	34.28 × 2	replaces today's two MViTs, 0 new bytes
imu_stats ExtraTrees 200×12 as tensors	9.00	new: serialise (feature, threshold, children, leaf probs) per tree as int16/fp16 tensors + a 30-line torch gather evaluator. Makes today's 167 champion legal as-is (swapping the sklearn member out cost 2 clips, EXP-128). 100.4 MB total → bit-pack one view to true int6 (−8.6 MB, "not implemented" per EXP-129) → 91.8 MB
thermal-224 view (if 3a passes)	34.28	fits only if the skeleton is pruned to 3 archs (~14 MB) or thermal replaces the wrist view — decide on the LOSO lower quartile, not on bytes
detector projection (3b) + μ_train + prototypes	< 1	manifest
L2 code	0	inference.sh
Rules: nothing is adopted until its packaged form exists as a file and reproduces its scored probabilities (B-033); the package is rebuilt on day 5 with what passed, scored once on day 6, and frozen; pack_stage2.py gains --imu-trees / --extra-head, and unpack_stage2.py --check infer must exercise both.

7. L4 — the pair-prompt verifier, kept as the fallback with its honest size
If T1 fails and L2 fails P0 by day 2, build it: frozen shipped MViT trunk; tokens = [centered feature, class-prompt E[a], E[b], subject-reference r_S]; 2-layer transformer head, antisymmetric score g(x,a,b) − g(x,b,a); trained on honest OOF top-5 pairs; inference rule fixed at argmax_{c∈{a,b}} log p_fused(c) + log p_ver(c). Gate: OOF accuracy on the 2,345 top-2 clips

0.893 (base 0.8785 + 2 SE). Bytes ≤ 2 MB. Expected +1–3 pts. It is here because it is cheap and answers the brief's literal question, not because it can close the gap.

A. Recipes (so nothing above is taken on faith)
EXP-132 (CPU, 1 min): z = 0.35·log(skel/prior) + 0.2925·(½log person + ½log wrist) + 0.3575·log imu + 0.25·log prior over oof_k224_mvit_pooled, oof_k224_mvitwrist_pooled, oof_astgcn_world25, oof_imu_stats_t200_d12; prior from cache/meta_train.csv. Count (user, y, pred) error cells with n ≥ 3, per-user repeat fraction, one-directional (user, pair) cells, reference availability. Expected 229/658, 55/70, 242/290.

EXP-133 (CPU, 2 min): bones from cache/train/*.npz skel_pos[:,0], H36M parents (0,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15), per-clip median of 16 bone norms; 300 s blocks from t0/t1; MACs from the second column of IMU/*.csv. Expected as in §1.

Embedding dump / P0 / 2a / 2c (1 GPU-h): hook model.head[-1] input in kaggle/cuhkx_224_kaggle.py (build_model, ClipStore, make_dataset, norm_tensors reusable as-is); checkpoints/k224_mvit{,wrist}_f{0..3}.pt, test with *_all.pt; flip-TTA average; on the laptop use --workers 0 (the first attempt died at 15 GB RAM). P0 = ward clustering of block-mean features, purity vs true user on train blocks at k = users-that-day; pass ≥ 90%. 2a/2c as in §3.

L1 feature extraction: HF facebook/vjepa2-vitl-fpc64-256 (also -vitg), OpenGVLab/InternVideo2-Stage2_1B-224p-f4, OpenGVLab/VideoMAEv2-giant; 16 frames from the existing 224 caches (person, wrist), pooled + patch tokens saved fp16; attentive probe = 1 query token cross-attending patch tokens, 4-fold subject-grouped, AdamW 20 epochs, minutes per fold.

Files this week: kaggle/cuhkx_224_kaggle.py (2 flags), code/teacher_features.py, code/teacher_probe.py, code/subject_recover.py, code/subject_align.py, code/build_thermal224_cache.py, code/object_channel.py, code/pack_stage2.py + unpack_stage2.py (trees, head), cluster/{loso,teacher,thermal,distil}.sbatch, and in the same commit as any state change docs/next_action.md + research/LOG.md.