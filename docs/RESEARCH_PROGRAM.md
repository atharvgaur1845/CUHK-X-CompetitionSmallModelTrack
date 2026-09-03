# The research problem underneath the competition

**Written 2026-09-03.** A diagnostic pass over artifacts already on disk — no new
training. Every number below was measured in this session and the command that produced
it is in `research/LOG.md` EXP-124. Read `docs/next_action.md` first for campaign state;
this file is about *mechanism*, and about what changing what is predicted to do.

**Provenance discipline.** Each finding is marked **NEW** (not previously in the ledgers),
**SHARPENED** (known, now quantified on the *current* system), or **REDISCOVERED** (already
established — recorded so the next session does not spend hours re-deriving it, as I did).

---

## 1. The headline: this is a RANKING problem, not a recognition problem

**NEW.** The champion fusion on 2,700 pooled OOF clips:

| | accuracy |
|---|---|
| top-1 | 0.7630 |
| **top-2** | **0.8685** |
| top-3 | 0.9067 |
| top-5 | 0.9393 |
| top-10 | 0.9700 |

**Of the 640 errors, 285 — 44.5% — have the true label at exactly rank 2.** Resolving
*only* the rank-1-vs-rank-2 decision, changing nothing else, takes the system from
**0.7630 to 0.8685: +10.55 points**. Cumulatively, rank ≤3 covers 60.6% of errors and
rank ≤5 covers 74.4%.

The model is not failing to see the activity. It is failing to *order two candidates it
has already surfaced*.

### 1a. The oracle gap and the rank-2 gap are nearly the same set

**NEW, and it reframes T3.** Oracle-any-member over the five members is **0.8774**. The
champion's own **top-2 is 0.8685**. These are within 0.9 points of each other.

The campaign has described its headroom as *member selection* — "309 clips are winnable by
some member, and no global weight can reach them". The measurement says something
simpler and much more actionable: **almost every clip a different member would have won is
a clip the champion had at rank 2 anyway.** You do not need to know *which member* to
trust. You need one binary decision, on a candidate pair the fusion has already produced.

**What this predicts for T3 (EXP-122, queued).** A student distilled to imitate a 40-way
soft target spends most of its capacity on the 38 classes that were never in contention.
The oracle target is worth more than the fused target *because it sharpens exactly this
pair* — which is why `oracle − fused` is the contrast worth reading. And it suggests a
cheaper sibling experiment: **a binary head over (rank-1, rank-2) features**, which is a
far smaller learning problem than 40-way distillation on 2,148 clips.

---

## 2. Cross-subject variance dominates — and it *derives* the noise floor

**NEW.** Per-subject champion accuracy, 16 training subjects:

- best `user19` **0.8172**, worst `user23` **0.6515**
- **between-subject sd = 5.26 points**, mean 0.7611

Compare the other variance sources we have measured: **seed σ = 1.16** (EXP-120a), fusion
weight sweeps ±0.4 over 5,227 points. Subject identity is a **4.5× larger** source of
variance than anything the campaign has been tuning.

**This derives the noise floor that was previously only empirical.** The public split is
**4 subjects**. Sampling 4 subjects from a population with sd 5.26 gives
SE = 5.26/√4 = **2.63 points**, so a 2-SE band is **±5.3 points ≈ ±10.6 clips of 201** —
which is the ±9–10 clip noise floor `CLAUDE.md` states as an observed fact. The floor is
not measurement sloppiness; it is *subject sampling*, and it cannot be reduced by
running more seeds or averaging more members.

**Consequences that follow directly:**

- **The on-site stage (8 new subjects) has SE = 1.86 points** — narrower than public, but
  our score there is still a draw from a distribution with a 16-point spread between the
  easiest and hardest subject we have seen.
- **A +2-clip public margin is not a margin.** It is a fifth of one standard error.
- **Any method that reduces per-subject variance is worth more than the same expected
  gain in mean accuracy**, because the grade weights private (8 subjects) and on-site
  (8 subjects, 30%) at 2.5× the public leaderboard. This is an argument for subject-level
  normalisation and against anything tuned on 4 public subjects.

---

## 3. The error is broad, not pair-concentrated — which kills one obvious fix

**SHARPENED.** That the confusions are object-identity pairs is known
(`LOG.md:2245`: 6 `Drink_water` ↔ 7 `Eat_food`, 12 `Sweep` ↔ 13 `Mop`,
21 `Read_documents` ↔ 22 `Turn_pages`). What was not quantified is the *concentration*:

| | share of all 640 errors |
|---|---|
| inside a symmetric confusion pair | **49.7%** |
| top-10 unordered pairs | 32.3% |
| top-20 unordered pairs | 45.5% |
| spread over | **175 distinct pairs** |

Largest single pair: **21 `Read_documents` ↔ 22 `Turn_pages`, 43 errors — 6.7% of the
total.** Every one of the twelve worst classes is an **OBJECT** class.

**This is decisive against `QUEUE.md` X-02, "nested low-parameter pair specialists".** Ten
specialists buy access to 32% of the error mass and each is fitted on a few dozen clips.
The shape that fits the data is **one pair-conditioned discriminator** — a single model
taking (clip features, candidate A, candidate B) and emitting a preference — which sees
all 640 errors as training signal instead of splitting them 175 ways. Same hypothesis,
one model instead of twenty, and it is the same object as §1's binary re-ranker.

---

## 4. Recording structure: a lever whose SIGN depends on base accuracy

**REDISCOVERED, then SHARPENED into a live prediction.** I re-derived the recording
structure from timestamps and thought I had found something new. I had not — `EXP-097`
and the 2026-07 audit got there first. Recording it so the next session does not repeat
the detour:

- train blocks at a 60 s timestamp gap are **100% single-user** (740/740)
- **93.4%** of those blocks have all-distinct labels; consecutive clips share a class
  **0.1% of the time** (2 of 2,191)
- at *session* scale (5 min) distinctness is **false** — 1,937 duplicate-label clips
  (EXP-097 killed this, and my sweep reproduces it: gap<300 s scores **−39 points**)
- `ordered_transition_decoder.py` already implements `--distinctness none|hard|penalty`
  and already groups by radar recording pass

**What is genuinely open is B-022's standing prediction**, and it is now testable at a
base accuracy far outside the range where it was measured. B-022 (85% confidence) says
distinctness coupling pays *only above a base-accuracy threshold*: with a reliable
posterior the constraint propagates correct information; with an unreliable one it
propagates error. The public ladder:

| base | distinctness delta |
|---|---|
| 112 clips | **−1** |
| 121 clips | **0** |
| 123 clips | never run |
| **166 clips (today)** | **?** |

Monotone, as predicted, and never yet positive — because it has never been tried above
123. **On today's fusion I measure it positive for the first time:** soft distinctness
(repeat penalty λ=2.0, blocks at a 30 s gap) gives **+2.41 points — 102 clips rescued
against 37 broken**, on a smooth plateau across λ ∈ [1,4], not a knife-edge fit.

| variant | OOF | delta | rescued / broken |
|---|---|---|---|
| baseline argmax | 0.76296 | — | — |
| hard all-distinct, blocks ≤8 | 0.77815 | +1.52 | 85 / 44 |
| **soft penalty λ=2.0, gap<30 s** | **0.78704** | **+2.41** | **102 / 37** |
| soft penalty λ=2.0, gap<60 s | 0.76074 | −0.22 | 100 / 106 |

> ### ⚠ PREDICTION CORRECTED, same day. I predicted **+4 to +5 public clips. That was wrong.**
>
> The +2.41 above was measured against **raw argmax**. The champion already runs the
> transition decoder, which captures most of the same structure — consecutive clips in a
> recording almost never repeat a class, and a first-order Markov model learns that on its
> own. Run through the **real decoder** on top of the champion's own settings, the
> *incremental* value is **+1.00 point**, not +2.41:
>
> | | pooled OOF | rescues | harms |
> |---|---|---|---|
> | `--distinctness none` (champion) | 0.80148 | 191 | 87 |
> | `--distinctness penalty 2.0` | **0.81148** | **216** | **85** |
>
> **4/4 folds positive** (+1.48 / +0.74 / +1.03 / +0.74, mean +1.00, sd 0.35), and it adds
> 25 rescues while *removing* 2 harms — the ideal signature.
>
> **But on test it moves only 6 of 405 rows**, which the OOF change rate predicts exactly
> (43 extra changes on 2,700 = 1.59% → 6.5 rows on 405). So ~3 rows on the public 201, and
> an expected **+1 clip, range −3 to +3**.
>
> **The lesson generalises:** measure an add-on against *the system you actually ship*, not
> against argmax. Half the apparent gain was already being collected by a component that
> was switched on.

The mechanism still stands (the 30 s-vs-60 s sign flip is the constraint being true of
tight recording passes and false of sessions), and the change is protocol-grounded rather
than fitted — it should transfer to private and on-site better than the transition model,
which estimates statistics from train users. It is currently *off* in the champion recipe
(`--distinctness` defaults to `none`), and `submissions/sub_r2_dist.csv` is built.

---

## 5. Thermal: the stated mechanism was wrong (see EXP-123)

**NEW, and it corrected `CLAUDE.md`.** The ledgers said thermal contributes zero because
it is uncalibrated. Measuring the same confidence-separation statistic across all members
on identical clips, **thermal has the highest separation (+0.0650) and our best-fusing
member has a negative one (−0.0202)**. The criterion does not discriminate.

The supported reading is **accuracy**: thermal brings 19 rescues against 247 errors, so
any weight that harvests the 19 imports from a pool 13× larger. And the dataset paper
ranks thermal **first of six sensors (92.57)** while our member sits at **0.544** against
the IR+depth member's 0.715 — a modality that should be our best is our second-weakest.
Five of the six differences from the tied team's thermal recipe remain untested; the crop
(the one we tested) is not the defect.

---

## 6. What changing what is predicted to do

Signs and magnitudes are predictions from the mechanisms above, recorded **before** the
experiments so they can be scored honestly.

| change | predicted effect | mechanism | confidence | cost |
|---|---|---|---|---|
| **Turn on soft distinctness** | **+1 clip (range −3..+3)** — corrected down from +4/+5 | §4; +1.00 pt OOF 4/4 folds, but only **6 of 405 rows** change | 70% it is ≥0 | 1 submission |
| **Binary rank-1-vs-rank-2 re-ranker** | up to +10.5 pts available; realistically +2 to +4 | §1 — 44.5% of errors are one binary decision | 40% that ≥2 pts is reachable | 1–2 days build |
| **T3 oracle target beats fused target** | +1 to +3 over `distil_fused` | §1a — the oracle target's advantage *is* the rank-2 pair | 55% | already queued |
| **Per-pair specialists (X-02)** | ≈0 | §3 — 10 models buy 32% of error mass, each fitted on tens of clips | 75% it disappoints | — |
| **Subject-level test-time normalisation** | small on public, **larger on private/on-site** | §2 — subject variance is 4.5× seed variance and the grade weights 16 unseen subjects at 2.5× | 45% | 1 day |
| **More seeds / more members in the video slot** | ≈0 | measured: 4-fold bag (137 MB) = single all-train (34 MB) = 166 | 85% it is flat | — |
| **Thermal recipe change (2D, 8 frames, 8 epochs)** | unknown; judge on member accuracy first | §5 — the binding constraint is 0.544 accuracy, not calibration | — | minutes/fold |
| **Anything tuned on the 4 public subjects** | **negative in expectation** | §2 — SE 2.63 pts; fitting inside one SE band is fitting noise | 80% | — |

---

## 7. The three questions worth a research programme

1. **Is the rank-1/rank-2 decision recoverable from the inputs at all?** If a binary
   discriminator on the 285 rank-2 clips cannot beat chance, then the ceiling is the
   *representation*, not the head — and the honest conclusion is that the remaining error
   is irreducible without a better sensor model. Either answer is worth knowing, and this
   is the fastest way to find out. **All five members are wrong together on 331 clips
   (12.3%)** — that is the part no re-ranking can reach.
2. **How much of the 5.26-point subject spread is removable?** This is the quantity the
   grade actually rewards (private 8 + on-site 8 subjects at 2.5× the public weight) and
   the campaign has never attacked it directly. It is also the reason the public
   leaderboard has been such a poor guide.
3. **Why is thermal at 0.544 when the paper ranks it first at 92.57?** A 17-point gap
   between a modality's published strength and ours is the largest unexplained number in
   the project.

---

## 8. Method notes for whoever continues

- **Check the ledgers before believing a discovery.** Two of the findings above are
  rediscoveries that cost me an hour each. `LOG.md` is searchable; the graveyard in
  `OUTLIERS.md` exists for this.
- **A single-fold delta is not a result.** Seed σ = 1.16, so a paired single-fold delta
  carries 1.64 and the 2-SE bar is **3.28** (EXP-120a). Use paired multi-fold designs,
  where four folds bring the bar to 1.64.
- **The public split is 4 subjects with SE 2.63 points.** Treat any public delta under
  ~5 points as unmeasured, and prefer mechanisms that should generalise to unseen
  subjects over anything fitted to these four.
