# Figure audit, report v2 (2026-09-17)

Every figure is produced by `generate_all_figures.py` from the tables written by `figdata.py`
into `figures_src/*.json`. No number is typed into a figure by hand; the JSON files are the
plotted values. Style: `figstyle.py` (one palette, Lato, vector PDF, fonts embedded).

Semantic colours: person BLUE `#2E5C9E`, wrist BLUE_LIGHT `#6F97D1`, thermal ORANGE `#D9772B`,
skeleton/IMU TEAL `#1F7F7A`, IMU forest TEAL_LIGHT `#5FB3A9`, decoder/structure PURPLE `#6B4FA0`,
negative RED `#B4433A`, baseline GRAY `#8A929B`.

| # | file | claim | source experiment | source artifact | exact values used | visual encoding | interpretation risk |
|---|---|---|---|---|---|---|---|
| 1 | `fig01_abstract.pdf` | five streams, five specialists, fusion, decoder, one 96.37 MB file | EXP-162 harness (member OOF), EXP-158/159 (scores), manifest | `oof_*.npz`, `weights_p384.json`, `stage2_ship7.pth` manifest, one training clip `30_Do_jumping_jacks/user2/5-1-1` | member alone 1980/1944/1711/1668/1017 of 2700 (0.733/0.720/0.634/0.618/0.377); fused 2109 (0.781); decoded 2240 (0.830); bytes 21.6 x3 / 18.9 / 1.6 MB; private 0.877, 14/307; 0/405 across 5 GPUs | node diagram; real IR/depth/thermal frames, real skeleton joints (x, z), real IMU |acc| traces, real radar points; the "3 repeats" tile is the clip's own recording group (jog, squats, jacks x3) | the sample clip is illustrative of the modalities, not of a typical error; accuracies are OOF (held-out training subjects), the card's 0.877 is Kaggle private: both labelled |
| 2 | `fig02_complementarity.pdf` | members rescue different clips; every member load-bearing | EXP-162 harness; logged EXP-140/113 values differ (earlier stage) and are recorded in `members.json` | `members.json` | patterns: 11111 646/646, 11110 495/495, 00000 301/8, 11100 187/183, 11010 116/112, 11011 115/115, 11000 113/103, 00010 73/16, 00100 68/8, ... ; singles skeleton 73, thermal 68, person 61, wrist 52, forest 34; oracle 2399 (0.889), fused 2109 (0.781); LOO person -48, wrist -25, skeleton -23, thermal -12, forest -11 | UpSet-style bars + membership dots; lollipops | LOO here is at the shipped stage (EXP-162 "removed w=0"); the text of earlier sessions quoted EXP-140's 224-px-stage LOO (-53/-31/-17/-16/-2), which is a different configuration. The paper now uses the shipped-stage numbers everywhere |
| 3 | `fig03_resolution.pdf` | crop-size rule | EXP-154, EXP-160 | `resolution.json` | person 0.71444/0.72963/0.73333/0.72519; wrist 0.7200 to 0.7064 (fold deltas +0.25, -2.46, -2.91, -0.31); 448: member -0.81, fusion -17, t=-2.02; thermal 288 fold 2 -1.99, fold 3 +2.14 | connected points, direct labels; schematic pixel grids (16 vs 4 cells) | schematic box sizes are illustrative of the stated ranges (person 224 to 480 px, wrist about 96 px), not measured per clip; the y axis is truncated (70 to 74.6) but every point is labelled with its value |
| 4 | `fig04_thermal.pdf` | same camera, different reading | EXP-088/113 (crop 128), EXP-120b (full 128), EXP-139 (224), EXP-140 (public) | `thermal.json`; frames from `9_Pour_drinks/user2/2-1-1` | 0.544; 0.568 (folds 0.58354/0.56511/0.55828/0.56508); 0.627 (0.66216/0.63759/0.60583/0.59112); decoded fusion 2184 base, 2225/2226/2230/2229/2218 at w 0.10/0.15/0.20/0.30/0.40 (+41/+42/+46/+45/+34); public 167 to 172 | ladder with fold dots; gain curve; two real thermal frames | the "crop 128" inset is a centre crop of the frame chosen for illustration, not the detector window of that clip; 0.544 has no per-fold values in the log |
| 5 | `fig05_subjects.pdf` | fusion raises 14/16 subjects, narrows spread | EXP-141, EXP-145 | `loso.json` (parsed from LOG tables) | 16 paired subjects; person mean 0.7301, sd 0.0643, spread 0.2296; ensemble mean 0.7822, sd 0.0542, spread 0.2097; paired delta mean +0.052, min -0.0149 (user4), max +0.104 (user22) | slope pairs; mean/sd/range strip | EXP-145's headline compares 18-subject person stats (sd 6.9, spread 28) with 16-subject ensemble stats; the figure uses the paired 16-subject values and the caption states both. The skeleton contributes 4-fold OOF, not LOSO (understates the ensemble) |
| 6 | `fig06_robustness.pdf` | stability under missing sensors and coefficient scaling | EXP-162 (a), (b) | `robustness.json` | constant posterior -48/-29/-11/-27/-13; removed -48/-25/-12/-23/-11; coefficient grid person 2085/2102/2109/2114/2102, wrist 2102/2111/2109/2109/2107, thermal 2113/2116/2109/2107/2105, skeleton 2100/2104/2109/2107/2100, IMU 2105/2109/2109/2107/2105, prior 2109/2110/2109/2108/2111 | heatmap, linear diverging colour, every cell numbered | "constant posterior" is the member's mean posterior over clips, the closest label-free proxy for a zero-input pass (EXP-162 note) |
| 7 | `fig07_decoder.pdf` | decoder gain = distinctness + order; neutral without structure | EXP-162 (c) | `decoder.json` | base 2109; uniform 2164 (+55, 82/27); intact 2240 (+131, 187/56); shuffled 2118 (+9, 114/105); permuted 2092 (-17, 82/99); reversed 2055 (-54, 88/142) | waterfall; diverging rescue/harm bars | the decomposition order (distinctness first, then order) is a presentation choice; the two controls are measured independently and the sum 55 + 76 = 131 is arithmetic on measured totals |
| 8 | `fig08_landscape.pdf` | levers measured, bottleneck is not capacity | EXP-102/103/104/112/115/122/127/125/134/135/137/139/143/146/147/148/154/156/157/160 | `landscape.json` | one row per lever with delta, level, verdict, EXP id (see JSON) | dot plot by lane; circle = member points, diamond = system points (clips/27); off-scale arrows for -9.8 and -29.3 | mixed levels on one axis: the marker shape and legend state the level; the adoption bar is the member-level bar; system-level decisions used paired fold t and rowdiff readability (caption) |
| 9 | `fig09_bytes.pdf` | budget as constraint | package manifest; EXP-144/149/150/153 for the notes | `bytes.json` | video 64.79 (21.6 x3), skeleton 18.93, detector 10.30, forest 1.58, windows 0.01 + manifest 0.18; total 96.37; free 3.63 | stacked container to 100 MB | the three video members are shown as equal thirds of the video branch total (they are the same architecture and quantisation; per-member bytes differ by < 0.1 MB) |
| 10 | `fig10_verification.pdf` | drift exists, predictions unchanged | EXP-159, G6, G8, G14 | `gpu.json`, `docs/verification_logs/crossgpu/*/compare_vs_laptop.txt` | A100 1.6e-3 / 4.4e-5; H100 2.0e-3 / 4.3e-5; H200 2.0e-3 / 4.3e-5; V100 1.7e-3 / 9.2e-5; 0/405 each; CSV identical | matrix | the RTX 4060 row is the reference (dp = 0 by definition); its "0/405" is the docker run on the laptop vs the submitted file (G6) |
| 11 | `fig11_errors.pdf` | errors are subject-correlated and one-directional | EXP-132 (method), EXP-157 (twins, caption) | `errors.json` | 591 errors; 214 in repeated cells; 60 repeated cells; 56 subject-pair cells, 52 one-directional; top pairs Read/Turn 32, Sweep/Mop 19, Mobile/Games 14, ... | dot matrix (area = clips, colour = direction); per-subject bars | computed on the shipped fusion; EXP-132's logged 229/658 is for the earlier champion (stated in the caption); "3+" threshold follows EXP-132 |

## Numbers in the text that are not in a figure

| claim | source |
|---|---|
| 0.587 vs 0.627 prior-direction cost | early submission pair, CLAUDE.md conventions |
| 768-point grid, top-25 centroid, coefficients | EXP-154 |
| seed sd 1.16, bar 1.64 | EXP-120a |
| OOF vs public r = 0.91, bias 8 to 11 points | EXP-124 |
| public SE 6.9 clips (4 subjects), 8-subject SE 2.44 to 1.92 | EXP-141, EXP-145 |
| 10 test clips without thermal, 10 without a person detection | package run logs |
| 22 of 405 rows change if windows are recomputed on another GPU | EXP-159 |
| int8 detectors move 150 of 405 windows | EXP-149 |
| 403/405 primary reproduction, margins 0.012 and 0.006, about 900 coefficient settings | `docs/verification_logs/config_search_*`, README section 7 |
| timings 2 min 39 s (A100), 4 min 53 s (RTX 4060), 5.4 GB peak | EXP-159, G-measure log |

## Checks performed

- every figure compiled to vector PDF with embedded fonts (`pdffonts`: Identity-H, embedded); the only raster content is the real sensor frames in Figs. 1 and 4;
- every page rendered at 110 dpi and inspected for overlaps and label collisions (fixed: Fig. 2 callouts, Fig. 3 labels, Fig. 4 ladder labels, Fig. 5 strip labels, Fig. 6 headers, Fig. 8 label spreading, Fig. 9 label spreading, Fig. 10 headers, Fig. 11 label sizes);
- minimum text size in figures 5.5 pt (lane labels in Fig. 8, column labels in Fig. 11), body 7.5 pt;
- no em dashes in the report or the figures;
- caption numbers cross-checked against `figures_src/*.json` by script (see the session log);
- the previous report's five figures were compared with the new set: every quantity they carried is present in the new figures with the same values (resolution curve, LOSO per subject, missing-modality and coefficient tables, decoder controls, byte budget).
