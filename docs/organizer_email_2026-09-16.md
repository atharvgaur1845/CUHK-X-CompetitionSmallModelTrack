# Email to the organisers — reproduction of our primary Selected submission

**To:** cuhkx.competition@gmail.com
**Subject:** Small Model Track, team AGX — question on reproducing the primary Selected submission (ID 56212349)

Dear Organizing Committee,

Thank you for the Selection Stage documents. Before we finalise our verification package we want to
disclose one point and ask how you would like it handled.

Our two Selected submissions are ID 56212349 (`sub_pkgship6.csv`, private 0.87745, the primary
target) and ID 56213683 (`sub_final_ship7.csv`, private 0.87254). Both were produced by the same
frozen models — three MViTv2-S video members, a skeleton/IMU ensemble, an IMU tree model and two
YOLO detectors — fused with fixed coefficients and decoded with a fixed transition model.

The frozen package we will submit (one file, 96.4 MB) reproduces ID 56213683 row-for-row on the
Kaggle test set from raw frames, and we have verified this on two machines. For ID 56212349 the
package reproduces 403 of 405 rows. The reason is that the CSV for 56212349 was fused two hours
before 56213683 using the skeleton ensemble's full 48-member output; to satisfy the 100 MB rule the
packaged skeleton ensemble keeps 20 of those members (the ones carrying fusion weight). The two
differing rows (`SM_test_0206`, `SM_test_0382`) are near-ties in the fused posterior. Fusion
coefficients that make the packaged 20-member ensemble reproduce 56212349 exactly may exist; if we
find one we will ship it as the configuration for 56212349 with this explanation, and the
configuration for 56213683 unchanged, both selectable through the optional third argument of
`inference.sh` as your guide describes.

Our question: for the primary verification target, would you prefer (a) the package configured so
that it reproduces the submitted CSV exactly, with the coefficient difference documented, or
(b) the package in the configuration used for 56213683, with the 2-row difference on 56212349
documented as the expected tolerance in README section 7? We will follow whichever you prefer and
will disclose the situation in README sections 2 and 7 and in `submission_info.yaml` either way.

Nothing else about the frozen solution changes. No test labels, external data or network access are
involved.

With thanks,
[names], team AGX (Small Model Track)
