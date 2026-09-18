# Speaker notes (generated from slides.tex by extract_notes.py; edit the \talk blocks, not this file)

Each slide: a 15 to 40 s script, the one sentence to land, the likely expert challenge, the answer. Slide text is never read aloud.


## 1. HERO: FIVE SENSORS. ONE DECISION.

**Script.** This is one training clip, jumping jacks, seen by the depth camera. No face is recorded. The same instant reaches the system through five different lenses: infrared and depth, thermal, a tracked skeleton, wrist and body inertial units, and radar. Five small specialists each read one of those streams; their votes are fused, the order of the recording adds context, and out comes one decision: jumping jacks. Forty activities, one 96-megabyte file, people the model has never seen, 0.877 on the private leaderboard, and the same 405 predictions on every GPU we tested.

**One sentence.** One human action, seen five ways, fused into one decision by a system that fits in 96 MB.

**Likely challenge.** Why five sensors instead of one good camera?

**Answer.** Because the task forbids an ordinary camera, and because, as slide 3 shows, the five streams make different mistakes: 2,399 of 2,700 held-out clips are right for at least one specialist, and no single one reaches 0.75.


## 2. WHAT IF THE CAMERA NEVER SAW YOUR FACE?

**Script.** The setting is a home or a care room where an ordinary camera is unwelcome, because identity must not be recorded. So the sensors are ones that see a body, heat, joints and motion, not a face. The catch is threefold: forty fine-grained activities, only eighteen people to learn from, and a single file of at most 100 megabytes. And the score that matters most is measured on people the model has never seen.

**One sentence.** Privacy-preserving sensors, unseen people, a 100 MB budget: the constraints define the problem.

**Likely challenge.** Isn't an infrared camera just a camera?

**Answer.** It is a body-shaped image, not a face-resolving one, and the depth channel we lead with carries no identity at all; the benchmark's definition of privacy-preserving is the organisers', and every stream we use is one they provide.


## 3. ONE SENSOR SEES ONLY PART OF THE STORY.

**Script.** Each square here is one held-out clip. Navy: all five specialists get it right. Grey: some do. The coloured stripe is the interesting part: 288 clips that exactly one specialist gets right, and every colour is there. The skeleton alone rescues 73, thermal 68, the body view 61, the hands 52, even the motion forest 34. Add them up and at least one specialist is right on 2,399 of 2,700 clips, 0.889, while the best single one is at 0.733. The information is in the members; the problem is selecting it. And removing any member costs the fusion between 11 and 48 clips.

**One sentence.** Different sensors make different mistakes, so the union is far larger than any one of them.

**Likely challenge.** Why keep the weakest member, the forest at 0.377?

**Answer.** It is the only member right on 34 clips, removing it costs 11, and its paired per-subject effect is neutral at worst; it also costs 1.6 MB.


## 4. FIVE SPECIALISTS. ONE DECISION.

**Script.** Here is the whole system in one motion. The body view and the hand view are two MViTv2-S video models on the infrared and depth crops; the heat view is the same architecture on the thermal full frame; posture is a twenty-member skeleton and inertial ensemble; motion is a small tree forest on wrist statistics. Their posteriors are combined as a weighted sum of log-probabilities, the coefficients chosen by nested cross-validation, which takes 2,700 held-out clips from 0.733 for the best member to 0.781. Then the ordered transition decoder uses the order of a recording session, which the radar timestamps give us, to reach 0.830.

**One sentence.** Complementary small specialists plus the structure of the recording, not model size.

**Likely challenge.** Why fuse in log space with a prior correction?

**Answer.** The video members are trained with a prior-corrected loss and emit uniform-prior posteriors, the skeleton ensemble emits train-prior posteriors; we divide the latter by the prior before summing. Getting the direction wrong cost four points on an early submission: 0.587 versus 0.627.


## 5. MORE PIXELS ARE NOT ALWAYS MORE INFORMATION.

**Script.** The person crop climbs from 0.714 at 224 pixels to 0.733 at 384, four folds out of four, and falls back at 448. The wrist crop does the opposite: from 224 to 288 it loses 1.36 points, three folds out of four. The reason is on the right. A person box spans 224 to 480 source pixels, so a 384-pixel input samples detail the frame really has. A wrist box is about 96 source pixels, so at 288 each source pixel becomes nine input pixels of interpolation. More pixels, no more information.

**One sentence.** Resolution is crop-dependent: it pays where the crop contains real detail.

**Likely challenge.** Did you test beyond 384?

**Answer.** Yes: 448 is measured, minus 0.81 on the member and minus 17 fused clips with t = minus 2.0; not shipped.


## 6. THERMAL WAS USELESS. UNTIL WE READ IT CORRECTLY.

**Script.** Thermal is the dataset paper's strongest single sensor, yet our first thermal member, a 128-pixel person crop through a 3-D CNN, scored 0.544 and added exactly nothing in fusion at any weight. Reading the full frame at 128 pixels gave 0.568, still not adopted. Reading the full frame at 224 pixels through the same MViTv2-S recipe as the video members gave 0.627, plus 5.6 points on four folds out of four, and for the first time thermal helped the system: 46 more decoded clips out of 2,700 and 5 more public clips. The camera never changed. The 320 by 240 frame is read near-natively, and the objects and the room stay in view.

**One sentence.** Thermal became a contributing sensor when it was read at native resolution with the objects in view.

**Likely challenge.** Why was it useless before, if the sensor is strong?

**Answer.** Wrong recipe for the sensor: cropping removed the objects that define most classes, and 128 pixels threw away most of the frame. Decorrelation alone did not make it useful; accuracy did.


## 7. THE REAL TEST IS THE PERSON YOU'VE NEVER SEEN.

**Script.** Subjects, not clips, are the unit of generalisation here, so we hold out one person at a time. Grey is the person view alone; navy is the shipped ensemble; each vertical line is one held-out person. Fourteen of sixteen go up, the mean goes from 73.0 to 78.2, and the spread between the hardest and the easiest person narrows from 23 points to 21. The two that go down are within seed noise. Fusion does not only raise accuracy; it makes the system more even across people, which is what the held-out evaluation rewards.

**One sentence.** Fusion raises fourteen of sixteen held-out people and makes them more alike.

**Likely challenge.** Leave-one-subject-out is not a new site.

**Answer.** Agreed: it is sixteen draws of an unseen person, the closest measurement we have; a draw of eight subjects still carries about two points of standard error, which is why we do not oversell the leaderboard either.


## 8. WHAT HAPPENS WHEN A SENSOR DISAPPEARS?

**Script.** This is a reliability test, not an ablation for its own sake. Remove any sensor at inference, the way the pipeline really handles a missing stream, and the worst case is the body view: 48 clips of 2,700, 1.8 points. Thermal and the forest cost eleven to twelve. Scale any fusion coefficient by half or by one and a half, and the fused accuracy stays within minus 24 to plus 7 clips, under one percent. No sensor is a single point of failure and the coefficients are on a plateau.

**One sentence.** No single point of failure; the coefficients sit on a plateau.

**Likely challenge.** Did you fit the weights to the leaderboard?

**Answer.** No: nested cross-validation inside the folds; and the plateau shows it would not matter, at most 8 of 405 rows move under any re-weighting.


## 9. THE RECORDING ITSELF KNOWS SOMETHING.

**Script.** The fused argmax gets 2,109 clips right. The first thing the recording tells us is that its activities are distinct: a soft penalty on repeating a label within a session adds 55. The second is the scripted order: a first-order transition model over classes, fitted on training metadata, adds 76 more through beam search. That is 2,240, plus 131 clips, and the same decoder gave plus three to five clips on the public leaderboard. The controls on the right are what make this science rather than a trick: shuffle the order within sessions and the gain is plus 9, neutral; reverse it and it is minus 54; permute the class labels and it is minus 17. The decoder is reading structure, not memorising label frequencies.

**One sentence.** Recording structure carries measurable information, and the controls show it is structure, not memorisation.

**Likely challenge.** You are exploiting the recording order.

**Answer.** It is label-free, fitted on training metadata only, and neutral when the order is absent: a clip without a radar key simply gets the fused argmax.


## 10. SO WE TRIED THE OBVIOUS THINGS.

**Script.** We tried the obvious things first. More capacity: VideoMAE-B with 87 million parameters lost 1.8 points on zero of four folds, Swin3D-T lost 9.8, a V-JEPA 2 teacher was 29 points behind. More training tricks: layer-wise decay was noise, distillation and self-training helped a member and hurt the fusion. More test-time tricks: temporal augmentation, seed soups, a pair verifier, all negative or unreadable. Resolution was mixed, and that is the honest version: 288 and 384 on the body helped, 448 and the wrist step did not. What worked was a different kind of information: a new sensor view, fusing in the right space, and the structure of the recording.

**One sentence.** More of the same did not help; a different kind of information did.

**Likely challenge.** Isn't MViTv2-S itself a capacity upgrade?

**Answer.** It replaced a similar-sized 3-D CNN, plus 2.4 points at 34 million parameters; the larger models above it all lost. Capacity beyond that point overfits 2,281 training clips.


## 11. WE FIT AN ENTIRE SYSTEM INTO 100 MB.

**Script.** The container is the 100-megabyte rule. Three video models at five-bit weight codes take 64.8 megabytes, two thirds of it. The skeleton and inertial ensemble, pruned from 48 members to 20 and stored at int8, takes 18.9. Both detectors, 10.3 megabytes at fp16, plus the forest, the crop windows and the manifest. 96.37 megabytes, 3.6 to spare. Two decisions were measured, not assumed: five-bit video cost one public clip against six-bit, recovered elsewhere; and quantising the detectors to int8 would have moved 150 of 405 crop windows, so they stayed at fp16.

**One sentence.** Everything the system needs at inference lives in one 96.37 MB file, and every byte was a measured trade-off.

**Likely challenge.** Where are the detectors?

**Answer.** Inside the file, fp16, 10.3 MB; the 405 crop windows are stored too, so the Kaggle run is reproduced exactly, and new data goes through the live detectors.


## 12. AND IT ACTUALLY RUNS.

**Script.** The package is one command: no arguments beyond the data directory and the output path, no network, the data mounted read-only. We verified it from the actual uploaded archives in a clean room, twice, with the working repository moved aside. Then we ran the same container on five GPU generations. Posteriors drift by up to two in a thousand, the way half-precision arithmetic does; the argmax changes on zero of 405 clips and the CSV is byte-identical every time. That is the claim: not bitwise determinism, but a decision that does not move.

**One sentence.** Numerical drift exists; the decision does not move, on five GPU generations, from the uploaded archives.

**Likely challenge.** Your primary submission reproduces 403 of 405, not 405.

**Answer.** Yes, and we say so. The skeleton ensemble was pruned from 48 to 20 members after that submission to meet the size rule; two decoder near-ties flip. About 900 coefficient settings were tried and none closer, so we ship the principled coefficients and disclose the two rows rather than fit to the target file. The second submission is byte-identical.


## 13. TAKEAWAY (navy)

**Script.** Five sensors, one decision. Ninety-six megabytes, forty activities, people the model has never seen, 0.877 private. The residual error we are left with is per-subject and correlated: twin recordings of the same person share the identical wrong label 68 percent of the time, and no lever that assumes independent evidence reaches it. So the next gain is more subjects, not more model. We froze the competitive solution, reproduced it, stress-tested it, and measured what actually mattered. Thank you.

**One sentence.** Complementary small specialists plus structure beat model size, and the limits are measured too.

**Likely challenge.** What would you do with a month?

**Answer.** Subject diversity in representation learning, tested the same way: paired folds, a two-standard-error bar, and a per-subject delta, because a frozen-feature probe saturates at six subjects and any gain must come through the representation.


## 14. B1  THE SYSTEM, IN FULL

**Script.** The report's first figure: every sensing stream on the left as this clip sees it, the five specialists with their size and accuracy, the fusion, the decoder, and the result card.

**One sentence.** The whole system on one page.

**Likely challenge.** How large is each member?

**Answer.** Three MViTv2-S at 21.6 MB each in 5-bit codes, the 20-member skeleton/IMU ensemble at 18.9 MB in int8, the forest at 1.6 MB; the detectors 10.3 MB in fp16.


## 15. B2  FUSION: THE RIGHT PROBABILITY SPACE

**Script.** The fusion is a weighted sum of log posteriors with one twist: the members live in different prior spaces, and the skeleton ensemble must be divided by the training prior first. The coefficients come from nested cross-validation and sit on a plateau.

**One sentence.** Fuse in the right prior space; the coefficients are a plateau, not a fit.

**Likely challenge.** Why not learn a fusion network?

**Answer.** Every learned selector we tried, a verifier, alignment, an object probe, failed to recover the oracle's 2,399 from the posteriors; with 2,700 clips a six-coefficient plateau is the honest amount of fitting.


## 16. B3  WHY FIVE MEMBERS

**Script.** The full pattern table behind slide 3: 646 clips right for everyone, 301 wrong for everyone, 288 right for exactly one member; and the leave-one-out bars, person minus 48 down to the forest minus 11.

**One sentence.** Every member is load-bearing and the union is far larger than any member.

**Likely challenge.** Why not more members?

**Answer.** Adding members to the video slot is a measured graveyard: four person folds and one person plus one wrist both reach the same public score; the budget is better spent on a new view than on a fifth copy of an old one.


## 17. B3b  THE SKELETON / IMU ENSEMBLE

**Script.** The posture specialist is not one model but twenty: five families, each trained on four subject-grouped folds, on the skeleton track and the inertial units you see here. Stored at int8 it is 18.9 megabytes, a fifth of the budget.

**One sentence.** Twenty small models on the skeleton and inertial streams, 18.9 MB.

**Likely challenge.** Why an ensemble here and single models for video?

**Answer.** The skeleton members are small, so twenty of them cost 19 MB, and their fold diversity is what makes the ensemble's posterior calibrated enough to fuse; a video member is 21.6 MB each, so there the budget buys views, not seeds.


## 18. B4  THE FULL EXPERIMENT LANDSCAPE

**Script.** Everything from slide 10 plus the levers that did not fit there: the backbone change, the full-frame IR duplicate, the COCO object probe, twin pooling, subject alignment.

**One sentence.** Twenty-five levers, one bar, one map.

**Likely challenge.** Why so many failures?

**Answer.** They are the map; each was measured against a two-standard-error bar and recorded with its experiment id in the log.


## 19. B5  SUBJECT-LEVEL DETAIL

**Script.** The same slope pairs with the distribution strip, and the standard error of an eight-subject draw, which falls from 2.4 to 1.9 points.

**One sentence.** The dispersion falls as well as the mean.

**Likely challenge.** Are the two negative subjects a problem?

**Answer.** Minus 1.5 and minus 0.5 points are inside seed noise of 1.16 per model; the paired mean delta is plus 5.2.


## 20. B6  DECODER CONTROLS

**Script.** The decoder's gain decomposed and its controls with rescued and harmed counts: intact 187 rescued, 56 harmed; shuffled 114 and 105; reversed 88 and 142; permuted 82 and 99.

**One sentence.** The gain is structure, and the controls prove it.

**Likely challenge.** Would it work without radar?

**Answer.** Without a session key the decoder is off by construction and the system returns the fused argmax, 0.781 OOF instead of 0.830.


## 21. B7  THE PACKAGE

**Script.** What the organisers received and how it was checked: the archive, the image, fourteen gates ending in a clean-room run from the uploaded files with a comparer that shares no code with the package.

**One sentence.** Verified from the actual upload, not from the working tree.

**Likely challenge.** What if the held-out data reuses the Kaggle clip ids?

**Answer.** Crop windows are gated by a content fingerprint: a clip whose frames match the Kaggle clip uses the stored window, anything else goes through the live detectors; caches are per-run and wiped.


## 22. B8  LIMITATIONS AND DISCLOSURE

**Script.** Where the errors are and why they are hard: they cluster by subject and by confusable object classes, read versus turn pages, sweep versus mop. The disclosure on the primary submission is here in full.

**One sentence.** The limits are measured and disclosed, not hidden.

**Likely challenge.** Why not fit the two rows?

**Answer.** Because that would be fitting to the target file; the honour declaration and the verification guide both rule it out, and the two rows are worth less than the credibility of the package.
