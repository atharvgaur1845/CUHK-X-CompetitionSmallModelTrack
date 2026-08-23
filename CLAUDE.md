# CUHK-X Challenge — Small Model Track (UbiComp 2026)

> ## ▶ Start here — `docs/next_action.md`
>
> **Rule of this repo.** `docs/next_action.md` is the handover and the default task for
> any new session — a different model, a different tool, or a collaborator. Read it before
> planning work.
>
> **The reciprocal obligation:** any change that alters project state updates
> `docs/next_action.md` in the *same commit*. A stale handover is worse than no handover —
> it sends the next session confidently in the wrong direction.
>
> **Currently: best LEGAL score 0.79601 = 160/201** (`sub_q4`, a 66.1 MB package) —
> **exactly the top-15 qualification bar, with zero margin.** The 0.82587 = 166/201
> `sub_n8` result is **not a legal solution**: R-6 caps the whole inference package at
> 100 MB and that pipeline needs ~309 MB. Treat 166 as the ceiling to reach, never as
> our position. Standing target 0.89 = 179/201.
>
> **The fusion-weight, decoder, and bag-composition axes are exhausted** (see the
> "Do NOT" table in the handover); gain must come from stronger members — or from
> fitting more members under the byte budget, which R-6 explicitly encourages
> ("fp16 / int8 **or lower**").
>
> **A ">10% Kaggle-vs-package gap" allowance is cited in older notes and is UNSOURCED.**
> It appears in neither `RULES_VERIFIED.md` nor `OBJECTIVE.md`. Treat 100 MB as hard.

## What this is

40-class human activity recognition from privacy-preserving sensors, cross-subject.
201 public / 204 private test clips (405 total), 2,933 training clips from 18 subjects.
Metric: top-1 accuracy. Final score is Kaggle private 20% · **on-site test with 8 new
subjects 30%** · reproducibility 10% · report 20% · presentation 10% · efficiency 10% —
so **cross-subject robustness is worth 2.5× the public leaderboard.**

## Pipeline

```
Small-Model-Track/           →  code/build_crop_cache.py     YOLO11n person crop, IR+Depth
  Training/ Testing/            code/build_thermal_cache.py  same, thermal (3ch)
                             →  cache/crop_v1/    (2933+405, 16×4×128×128 uint8 memmap)
                                cache/thermal_v1/ (2933+405, 16×3×128×128)
                             →  code/train_video_crop.py     r2plus1d_18 Kinetics-400
                                code/imu_stats_member.py     ExtraTrees on 545 features
                                (skeleton stack `world25`, 48 members, pre-existing)
                             →  code/fuse_general.py         geometric fusion in log space
                             →  code/ordered_transition_decoder.py   Markov beam decode
                             →  submissions/*.csv
```

## How to run

```bash
python3 code/build_crop_cache.py --split train         # ~25 min CPU, YOLO on CPU
python3 code/train_video_crop.py --tag X --fold-oof oof_visual_mil_v1_f2.npz --resume
python3 code/infer_video_crop.py --tag X               # add --cache thermal_v1 for 3ch
python3 code/fuse_general.py --visual A.npz B.npz --visual-weights 0.45 0.55 \
        --weight 0.65 --output fused.npz
python3 code/ordered_transition_decoder.py test --probs research/artifacts/fused.npz \
        --transition-weight 0.5 --transition-score conditional --backoff unigram \
        --output submissions/sub_X.csv
python3 code/rowdiff.py submissions/sub_champion.csv submissions/sub_X.csv
```

Long runs go under `code/run_with_watchdog.sh <log> 'OUTER-ONCE' <cmd…>` — it kills and
resumes on a stalled log or 15 minutes of 0% GPU. **EXP-064 hung for twelve hours with the
process alive and the GPU idle; a crash announces itself, a deadlock does not.**

## Conventions

- **Measure changes by rowdiff, not by intuition.** The public noise floor is **±9–10
  clips**. A candidate that changes fewer than ~20 of 405 rows cannot produce a readable
  result — build it only if it is free.
- **One change per submission.** This has cost real information at least twice (EXP-080
  carried five differences; `vis4_055` changed members *and* weight), and a confounded
  submission is a wasted one at ±9 clips of noise.
- **Report first-run numbers separately from post-fix numbers.** The post-fix figure is a
  ratchet and is never a generalisation estimate.
- **Label every score's provenance** — first-run / ratchet / fitted / external oracle.
  Only Kaggle scores are external. OOF is biased high and has failed as an *adoption*
  criterion repeatedly; it is reporting-and-screening only.
- **Report rescued and broken, not net.** A fusion change that fixes eight clips and
  breaks two is not +8.
- **Verify the member's prior space before fusing.** Members trained with
  `cross_entropy(logits + log_prior)` (video, visual) emit **uniform-prior** posteriors;
  plain-CE members (skeleton) emit **train-prior** posteriors. `fuse_*.py` divides the base
  by the train prior to reconcile. Getting this backwards scored 0.58706 vs 0.62686 — the
  direction is measured, not assumed.
- **Never ship a tree member unsmoothed.** `imu_stats` emitted **2,382 exact-zero cells
  across 392/405 rows**; a zero in a geometric mean vetoes that class for the whole
  ensemble. That is the mechanism behind an earlier 0.29 submission. Laplace smoothing plus
  an assertion is mandatory.
- **Enumerate test clips with an explicit glob** (`TEST_ROOT.glob("SM_test_*")`). A stray
  `.claude` directory was once ingested as clip 0 and would have shifted every row against
  the submission order.
- **Adding an argparse field to a trainer invalidates `--resume`**, because the fingerprint
  is `vars(args)`. Never do it while a job is in flight; plain kwargs on helper functions
  are safe.
- **`OBJECT` is 31 classes** — `set(range(28)) | {37,38,39}` — not 26. Every "26 sedentary
  classes" figure in older notes is wrong.
- Long jobs must be resumable and must write per-epoch state. A 30-epoch video run OOM-died
  at epoch 24 with no checkpoint and cost 2.7 hours; optimizer state is deliberately *not*
  persisted because the 501 MB write killed the process.
- System RAM is 15 GB against a 3 GB memmap. `--workers 0`, `pin_memory=False`. A
  192×192 cache (6.9 GB) will thrash and is not currently viable.

## Rules that are verified, not assumed

`research/RULES_VERIFIED.md` holds the organiser rulings with their discussion-topic ids.
The two that matter most:

- **R-1: pretrained CNNs are legal.** "Small, standard pretrained CNNs such as
  ImageNet-pretrained ResNet18 are perfectly acceptable in the Small Model Track" — the
  restriction targets LLMs and large VLMs. **A prior session recorded the opposite as fact
  and cancelled the pretrained probe; months of SSL work existed as a substitute for a
  legal initialisation.** This is the single most expensive error of the campaign, and it
  was found in ten minutes by reading the API instead of reasoning.
- **L-1: a test-label leak exists and must not be used.** Confirmed, and permanently
  closed — see the handover's "Do NOT" table. All 30 subjects are accounted for; any
  subject not already in training **is** a test subject.

## Docs

Read-first order: **`docs/next_action.md`** → `research/DIRECTIVE.md` →
`research/LOG.md` (newest first) → `research/RULES_VERIFIED.md` → `research/BELIEFS.md`.

- `docs/next_action.md` — **the handover.** Default task; update in the same commit as any
  state change
- `research/DIRECTIVE.md` — the standing research directive (no premature ceilings;
  investigate external evidence before ideating)
- `research/RULES_VERIFIED.md` — organiser rulings with topic ids
- `research/LOG.md` — evidence log, newest first, verdict in every title
- `research/BELIEFS.md` — load-bearing beliefs with confidence and falsification tests
- `research/EVIDENCE.md` — evidence ledger, retracted claims kept visible
- `research/OUTLIERS.md` — hypothesis tournament and graveyard

## Transferable lessons from this campaign

- *Decorrelation is not sufficient for a fusion gain — the member must also be calibrated.*
  Thermal is maximally decorrelated (54% agreement) and strong (0.544) and contributes
  exactly zero, because its confidence when right ≈ its confidence when wrong.
- *An estimate is only valid on the distribution it was measured on.* OOF is positively
  correlated with public (+0.910) but biased high by 8–11 points.
- *Our own failure is not a ceiling.* "0.65–0.73 achievable" was asserted here and
  falsified by a live leaderboard showing 0.91542. The published notebooks were reachable
  the whole time.
- *Check the rule before building the workaround.* See R-1 above.
