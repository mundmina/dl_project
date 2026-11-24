# Paper plan

**New framing (2026-11-24, after stage 0): "Cue or distractor? Two kinds of background dependence and how to tell
them apart."** The remedy itself (swap, automatic masks, Waterbirds) is already published (see
`paper/related_work.md`); the contribution is the diagnostic, its validation on a controlled cue, the failure of
Grad-CAM to separate the two kinds, and which remedy fixes which kind.

**Original working thesis.** Models do not use the background as a cue; they are distracted by unfamiliar scenes.
Heat maps (Grad-CAM) miss this, probing and attribution show it. Swapping in real scenes with automatic
masks removes the problem at no accuracy cost and transfers to Waterbirds.

| Stage | What | Compute | Status |
|---|---|---|---|
| 0 | Literature and positioning: background augmentation / bias (Xiao et al. 2021 and related), unreliable saliency maps (Adebayo et al. 2018), Waterbirds methods (Group DRO, JTT, DFR). Decision point: if already published, narrow to "Grad-CAM vs probing/attribution". | none | **done 2026-11-24** — swap + auto masks + Waterbirds already published (ForAug, AutoBackSwap); proposed narrowing to "cue vs distractor", see `paper/related_work.md` |
| 1a | Attribution: region Shapley (all models) and Integrated Gradients — `08_attribution` | Colab, done | **done** (report §attribution) |
| 1b | Mask robustness: B trained with DeepLabV3 / dilated / eroded / box masks — `09_mask_robustness` | Colab ≈ 2.5–3 h | ready to run |
| 1c | Controlled cue: home scene per breed with strength ρ ∈ {0.5, 0.9, 1.0}; diagnostic + Grad-CAM — `10_cue_or_distractor` | Colab ≈ 2–2.5 h | ready to run |
| 2 | Waterbirds by the standard: ResNet-50; ERM, Group DRO, DFR; ours without group labels and ours + DFR | Colab | to do |
| 3 | ImageNet-9 Backgrounds Challenge (Xiao et al.); iWildCam only as a bonus | Colab | to do |
| 4 | Writing | none | to do |
