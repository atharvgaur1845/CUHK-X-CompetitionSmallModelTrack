# Draft email to organizers (Q-65) — FOR ATHARV TO REVIEW AND SEND

To: cuhkx.competition@gmail.com
Subject: [Small Model Track] Clarification on pretrained weights and model packaging

Hi CUHK-X Challenge organizers,

I'm competing in the Small Model Track and would appreciate written clarification on three rule points:

1. **Pretrained weights.** The rules prohibit "large pretrained backbones". Is ImageNet-pretrained initialization of *small* CNN backbones (e.g. ResNet-18, MobileNetV3, <25 MB) permitted, provided the final model stays within the 100 MB limit? Or must all weights be trained from scratch on the provided data? (For reference, the CUHK-X dataset paper itself fine-tunes ImageNet-initialized ResNet-50 baselines.)

2. **Model-size accounting for ensembles.** Does the 100 MB limit apply to the sum of all weights used at inference (i.e. an ensemble's total), and must everything be packaged in the single `checkpoints/model.pth` file mentioned in the verification instructions?

3. **Test-time processing.** Is transductive processing of the *unlabeled* test set inside the submitted inference code permitted (e.g. batch statistics adaptation over test inputs), given no test labels and no manual labeling are involved?

Thank you!
Team: <TEAM NAME — must match official website registration>

---
*Rationale: B-005/B-006 in BELIEFS.md — a ruling either way changes the visual-stream recipe (from-scratch vs finetune) and the ensembling budget. Answer expected to be conservative; plan assumes from-scratch until told otherwise.*
