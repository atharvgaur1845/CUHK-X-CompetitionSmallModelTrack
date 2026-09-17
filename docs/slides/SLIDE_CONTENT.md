# Slide content and plots (you build the deck; this is what goes on each slide)

Figures: `docs/slides/figures/*.png` (300 dpi) and `*.svg` (vector), generated from
`docs/report/generate_all_figures.py`. Every number below is in the report and in
`docs/report/FIGURE_AUDIT.md` with its experiment id. No em dashes anywhere.

Palette (use the same colours for any slide graphics): ink `#1B2A41`, person blue `#2E5C9E`,
wrist blue `#6F97D1`, thermal orange `#D9772B`, skeleton/IMU teal `#1F7F7A`, forest teal
`#5FB3A9`, decoder purple `#6B4FA0`, negative red `#B4433A`, gray `#8A929B`, band `#F3F1EC`.
Typeface: Lato (figures), any clean sans for slides.

Rule of the deck: one figure and one sentence per slide. The sentence is the title.

---

## 1. Title
**Five Sensors, One Model: Multimodal Activity Recognition Under a 100 MB Constraint**
CUHK-X Challenge 2026, Small Model Track, Team AGX. Atharv, BITS Pilani, K K Birla Goa Campus.
Visual: `fig01_abstract` cropped to the left column (the five sensor tiles) or the whole figure faded.

## 2. The problem in one line
**Recognise 40 everyday activities from sensors that cannot identify you, for people the model has never seen, with everything in one 100 MB file.**
- 2,933 training clips, 18 subjects; 405 test clips, 4 unseen subjects; a further held-out set of new subjects
- IR + depth (640 x 480), thermal (320 x 240), 17-joint 3-D skeleton, five inertial units, radar
- 40 classes, 31 involve an object; class prior 28:1
- scoring: private 20 %, held-out subjects 30 %, reproducibility 10 %, report 20 %, presentation 10 %, efficiency 10 %
Visual: the five sensor tiles of `fig01_abstract` (crop), or a table.
Threat: "why 100 MB?" It is the track rule, and it turned out to be an architectural constraint, not a nuisance (slide 12).

## 3. Thesis
**Under a 100 MB budget and cross-subject shift, accuracy comes from complementary small specialists plus structure, not from model size.**
- every larger single model we trained lost (slide 11)
- five small members, fused in the right probability space, decoded with the recording structure, beat them
- and they generalise more evenly across people (slide 9)
Visual: none, or the four key-findings tiles from page 1 of the report.

## 4. The system
**Five streams, five specialists, one decision.**
Visual: `fig01_abstract` full width.
Say: member accuracies 0.377 to 0.733 alone (OOF); fusion 0.781; decoder 0.830; one file 96.37 MB; private 0.877, 14th of 307.
Threat: "where are the detectors?" Inside the file (fp16, 10.3 MB).

## 5. Why five members
**Different members rescue different clips, and every member is load-bearing.**
Visual: `fig02_complementarity`.
- 2,399 of 2,700 clips are right for at least one member (0.889); the fusion gets 2,109 (0.781)
- 288 clips are right for exactly one member: skeleton 73, thermal 68, person 61, wrist 52, forest 34
- removing any member costs 11 to 48 clips
Threat: "why keep the IMU forest at 0.377?" It is the only member right on 34 clips and removing it costs 11; paired per-subject delta of dropping it is -0.25 points (EXP-145), neutral at worst.

## 6. Finding 1: resolution is modality-specific
**More pixels help the body crop and hurt the wrist crop.**
Visual: `fig03_resolution`.
- person: 0.714, 0.730, 0.733, 0.725 at 224, 288, 384, 448 px (4/4 folds up to 384; 448 negative, t = -2.0 fused)
- wrist: -1.36 points from 224 to 288 (3/4 folds negative)
- mechanism: the person box is 224 to 480 source pixels, the wrist box about 96; upsampling adds interpolation, not information
Threat: "did you try 448?" Yes, measured negative (EXP-160), not shipped.

## 7. Finding 2: the same sensor, read differently
**Thermal was useless at 128 px and became a contributing sensor at 224 px full frame.**
Visual: `fig04_thermal`.
- member 0.544 (crop 128) to 0.568 (full 128) to 0.627 (full 224, +5.6 points on 4/4 folds)
- fusion: 0.00 at every weight at 128 px; +46 decoded OOF clips and +5 public clips at 224 px
- 320 x 240 source read near-natively; objects and scene stay in view
Threat: "why was it null before?" Wrong recipe for the sensor: cropped, 128 px, 3-D CNN. The camera did not change.

## 8. Fusion and decoder
**Fuse in the right prior space; decode with the recording's structure.**
Visual: `fig07_decoder`.
- video members emit uniform-prior posteriors, the skeleton ensemble train-prior posteriors: the direction is measured (wrong way costs 4 points)
- coefficients from nested CV (top-25 centroid on a 768-point grid)
- decoder gain +131 clips = +55 from "distinct activities within a recording" + +76 from scripted order
- controls: shuffled order +9 (neutral), reversed -54, permuted labels -17
Threat: "you exploit recording order." Label-free, from training metadata only; neutral when the order is absent; permuting labels removes the gain, so it is not memorising class frequencies.

## 9. Finding 3: fusion narrows the subject spread
**The ensemble raises 14 of 16 held-out subjects and makes them more alike.**
Visual: `fig05_subjects`.
- mean 73.0 to 78.2; sd 6.4 to 5.4; worst-to-best 23 to 21 points (paired, 16 subjects)
- SE of an 8-subject draw (what private and held-out are) 2.4 to 1.9 points
Threat: "LOSO is not a new site." Agreed; it is 16 draws of an unseen person, the closest measurement we have; the two exceptions (-1.5, -0.5) are inside seed noise.

## 10. Finding 4: stable under perturbation
**No sensor is a single point of failure, and the coefficients are on a plateau.**
Visual: `fig06_robustness`.
- worst single-sensor loss 48 clips of 2,700 (1.8 points); thermal and forest cost 11 to 13
- every coefficient scaled x0.5 to x1.5 stays within -24 to +7 clips
- consequence: any re-weighting moves at most 8 of the 405 test rows
Threat: "did you fit the weights to the leaderboard?" No: nested CV; and the plateau means it would not matter.

## 11a. What we tried (one glance)
**Seven families of levers; three crossed out, four adopted, all by the same statistical bar.**
Visual: `slide_what_we_tried` (slide-only figure, `make_slide_figures.py`). Read the crosses first, then the checks. Say the numbers only if asked; they are on the slide in small type.

## 11b. What did not work, and why that is the result
**We tested the obvious routes; capacity was not the bottleneck.**
Visual: `fig08_landscape` (the same levers placed on one axis, at the level each was measured).
- larger backbones (VideoMAE-B -1.8, Swin3D-T -9.8), a foundation teacher (V-JEPA 2, -29), distillation (student 157 public vs 167), soups (-2.4), TTA (-4 fused), self-training (member up, fusion down), object detector (AUC 0.50), verifier (-0.84), twin pooling (-4 decoded)
- the adopted levers: resolution, a new view read correctly, prior-space fusion, recording structure
Threat: "why so many failures?" They are the map; each was measured against a 2 SE bar and recorded.

## 12. Fitting 100 MB
**The budget was an architectural constraint, not a compression exercise.**
Visual: `fig09_bytes`.
- 3 x MViTv2-S at 21.6 MB (5-bit codes), skeleton/IMU ensemble 18.9 MB (int8), detectors 10.3 MB (fp16), forest 1.6 MB, 3.6 MB free; 96.37 MB
- int5 video costs 1 public clip vs int6 (recovered by the thermal soup); int8 detectors would move 150 of 405 crop windows, so they stay fp16
Threat: "quantisation losses?" One clip, measured; the detectors are the sensitive part and were left alone.

## 13. Reproducibility
**Numerical drift exists; the predictions do not move.**
Visual: `fig10_verification`.
- one command, no network, data read-only; verified from the actual upload archives in a clean room, twice
- RTX 4060, A100, H100, H200, V100: posterior drift up to 2e-3, prediction changes 0/405, CSV identical
- second selected submission byte-identical; primary 403/405 with two disclosed decoder near-ties (skeleton ensemble pruned to fit 100 MB after that submission; about 900 coefficient settings tried, none closer; we ship the principled coefficients and say so)
Threat: "your primary needed a coefficient change." No. We disclosed two rows instead of fitting to the target file.

## 14. Results
**0.877 private (14th of 307), 0.866 public, 96.37 MB, 0.39 s per clip on an A100.**
Visual: the result card (Table I of the report) as a slide table.
- private 179/204; public 174/201 (four subjects; a public score carries about 7 clips of subject-sampling noise at 1 SE)
- inference 2 min 39 s for 405 clips on an A100, 4 min 53 s on a laptop RTX 4060, peak GPU memory 5.4 GB
- held-out result: to be added when published

## 15. Limitation and the direction it points to
**The remaining error is subject-correlated; more model is not the next lever.**
Visual: `fig11_errors`.
- 214 of 591 errors sit in a (subject, true, predicted) cell seen 3+ times; 52 of 56 such cells are one-directional
- twin recordings share the identical wrong label 68 % of the time
- more model: measured, no gain. More subjects: the direction the evidence points to, not a quantified prediction (a frozen-feature probe saturates at six subjects, so any gain must come through the representation)
Threat: "what would you do with a month?" Subject diversity in representation learning, tested the same way: paired folds, 2 SE bar.

## 16. Takeaway
**Five sensors, one model: complementary small specialists plus structure beat model size under a 100 MB budget, and the limits are measured too.**
Visual: `fig01_abstract` again, or the four findings tiles.

---

## Q&A bank (fact / interpretation / limitation), short form

- **Why MViTv2-S?** +2.42 over an IG-65M R(2+1)D on 4 folds (p = 0.0016); 224-native; 34.4 M parameters fit int5 in 21.6 MB. Limitation: no 3-D CNN ablation at 384 px.
- **Why not one large multimodal transformer?** 86.7 M parameters overfit 2,281 clips (VideoMAE-B 0/4 folds); the size rule; correlated modalities (full-frame IR+depth duplicated the crop). Limitation: no joint-multimodal transformer was trained.
- **Subject leakage?** Folds and LOSO are subject-grouped; test subjects unseen; the known test-label leak was closed and never used.
- **External data?** None. Kinetics-400 and COCO initialisations, declared.
- **Manual test inspection?** None; only counts and timestamps as features.
- **How do you know it generalises?** LOSO on 16 subjects; private rank equals public rank; flattening. Limitation: 4 to 8 subject draws carry 5 to 7 clips of noise.
- **Missing modalities at test time?** Zero-input path, identical to the Kaggle run (10 clips had no thermal); worst case 1.8 points on OOF.
- **Determinism?** fp16 autocast, fixed batches, no RNG; five-GPU table; the claim is unchanged predictions, not bit equality.
- **Weakest component?** The IMU forest (0.377 alone) and the object-class confusions (read vs turn pages, sweep vs mop).
