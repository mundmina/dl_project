# Paper plan

**Working thesis.** Models do not use the background as a cue; they are distracted by unfamiliar scenes.
Heat maps (Grad-CAM) miss this, probing and attribution show it. Swapping in real scenes with automatic
masks removes the problem at no accuracy cost and transfers to Waterbirds.

| Stage | What | Compute | Status |
|---|---|---|---|
| 0 | Literature and positioning: background augmentation / bias (Xiao et al. 2021 and related), unreliable saliency maps (Adebayo et al. 2018), Waterbirds methods (Group DRO, JTT, DFR). Decision point: if already published, narrow to "Grad-CAM vs probing/attribution". | none | to do |
| 1a | Attribution: region Shapley (all models) and Integrated Gradients — `08_attribution` | Colab, done | **done** (report §attribution) |
| 1b | Mask robustness: B trained with DeepLabV3 / dilated / eroded / box masks — `09_mask_robustness` | Colab ≈ 2.5–3 h | ready to run |
| 2 | Waterbirds by the standard: ResNet-50; ERM, Group DRO, DFR; ours without group labels and ours + DFR | Colab | to do |
| 3 | ImageNet-9 Backgrounds Challenge (Xiao et al.); iWildCam only as a bonus | Colab | to do |
| 4 | Writing | none | to do |
