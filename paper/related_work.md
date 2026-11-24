# Stage 0 — related work and positioning (2026-11-24)

Working thesis being checked: *models do not use the background as a cue but are distracted by unfamiliar scenes;
Grad-CAM misses it, probing and attribution see it; background swapping with automatic masks fixes it and transfers
to Waterbirds.*

## Verdict at the decision point

| Part of our thesis | Already published? | Closest work |
|---|---|---|
| Random background swap removes background sensitivity at no accuracy cost | **Yes**, many times | Tian et al. 2018; Xiao et al. 2021 (Mixed-Rand training); Chang et al. 2021; Ryali et al. 2021; ForAug 2025/26 |
| ... with automatic masks instead of human ones | **Yes** | ForAug (Grounded SAM, ImageNet scale); AutoBackSwap 2026 (patch detector from ~287 labelled images); Ryali et al. 2021 (saliency masks) |
| ... robust to mask errors (dilate / erode) | **Yes**, partly | ForAug Fig. 6: accuracy stable when masks grow, drops when they shrink; Ryali et al.: tolerant to mask noise |
| ... transfers to Waterbirds without group labels, competitive with Group DRO / DFR | **Yes** | AutoBackSwap: WGA 93.0 vs DFR 90.7, Group DRO 82.2 (ResNet-50); Chang et al. 2021 |
| Saliency maps miss differences that behavioural tests show | **Partly** | Moayeri et al. 2022: saliency alignment similar across models despite large differences in foreground sensitivity; Adebayo et al. 2022: post-hoc explanations fail to reveal unknown spurious signals. ForAug (App. E) reports the opposite direction: Grad-CAM / IG foreground focus rises with their augmentation |
| **Background as a cue (shortcut) vs background as a distractor**, measured separately | **Not found** | Closest: Xiao et al. 2021 (Only-BG accuracy = cue), Rosenfeld et al. 2018 (a transplanted object disturbs a detector = distraction, detection only), Moayeri et al. 2022 (relative foreground sensitivity, one number), "Seeing Through Distractions" 2026 (attribution methods credit distractors) |

**Decision: narrow the paper, as the plan foresaw.** The remedy (swap, automatic masks, Waterbirds) is not new and
AutoBackSwap already beats DRO/DFR on Waterbirds, so stage 2 as planned ("beat Group DRO without group labels") would
be a replication. What nobody seems to have done is to *separate the two kinds of background dependence* and show
which diagnostic and which remedy sees / fixes which kind.

## Proposed new framing

**"Cue or distractor? Two kinds of background dependence and how to tell them apart."**

1. **Diagnostic.** Per model, four numbers we already compute:
   - bg-only accuracy (can the background alone predict the class? cue);
   - region-Shapley φ_bg on the image's *own* background (> 0 = background helps = cue);
   - φ_bg on a *foreign* background (< 0 = new scene pushes away = distraction);
   - flip rate under swap.
   Pet ResNet-18 A: bg-only ≈ chance, own φ_bg +0.10, foreign −0.40 → **distractor**. Scratch A: own +0.74 → **cue**.
2. **Validation where the truth is known.** Controlled-correlation Pet: tie background scene to breed with strength
   ρ ∈ {0, 0.5, 0.9, 1.0}; the diagnostic must move from "distractor" to "cue" as ρ grows. Waterbirds ERM must read as
   "cue". This experiment is what makes the diagnostic credible — it is new work (≈ 12–15 ResNet-18 runs in Colab).
3. **Saliency cannot separate them.** Grad-CAM share-on-pet is the same for A and B (we have it); show it also cannot
   tell cue from distractor across ρ. Position against Moayeri 2022 / Adebayo 2022 and the opposite ForAug claim.
4. **Remedies act on different kinds.** Hypothesis to test: last-layer retraining (DFR) removes the *cue* but not the
   *distraction* (features still encode the scene — our probes), while background swap removes both. This would be a
   genuinely new result. Cheap: DFR is only a logistic regression on frozen features.
5. **Mask robustness (notebook 09)** stays as a supporting result, presented as consistent with ForAug on a
   fine-grained task with human masks as the reference.

Consequences for the plan: stage 2 becomes "Waterbirds as a known-cue case + DFR vs swap", not a leaderboard; stage 3
ImageNet-9 stays (Xiao's Only-BG / Mixed-Rand give a known mix of cue and distraction); add the controlled-ρ experiment.

## Notes per paper

- **Xiao, Engstrom, Ilyas, Madry — Noise or Signal: The Role of Image Backgrounds in Object Recognition** (ICLR 2021).
  ImageNet-9 with Only-BG, Mixed-Same, Mixed-Rand variants; BG-Gap = acc(Mixed-Same) − acc(Mixed-Rand); training on
  Mixed-Rand reduces background dependence. Code and data: github.com/MadryLab/backgrounds_challenge.
  [arXiv 2006.09994](https://arxiv.org/abs/2006.09994)
- **Nauen, Moser, Raue, Frolov, Dengel — ForAug: Mitigating Biases in Image Classification via Controlled Image
  Compositions** (arXiv 2503.09399, v4 July 2026, venue not stated). Grounded SAM masks + LaMa inpainting, paste the
  object on backgrounds from all classes; ImageNet-1k, ViT/DeiT/Swin/ResNet; reduces background, centre and size bias;
  mask erosion hurts, dilation does not; Grad-CAM/IG foreground focus increases (appendix). No Waterbirds, no probing,
  no cue-vs-distractor split. [arXiv](https://arxiv.org/abs/2503.09399)
- **Roder, Schweighofer — Automated Background Swapping for Robustness against Spurious Backgrounds (AutoBackSwap)**
  (arXiv 2606.32018, June 2026). Patch-level fg/bg detector trained on ~5% of Waterbirds, AOT-GAN infill, paste on
  random backgrounds, plain ERM, no group labels. Waterbirds ResNet-50 WGA/Acc: ERM 74.9/93.6, DFR 90.7/94.3,
  Group DRO 82.2/94.8, Chang et al. 85.7/95.7, AutoBackSwap 93.0/95.0; without minority groups AutoBackSwap 92.2 vs
  ≤ 79.1 for everything else. No interpretability analysis. [arXiv](https://arxiv.org/abs/2606.32018)
- **Chang, Adam, Goldenberg — Towards Robust Classification Model by Counterfactual and Invariant Data Generation**
  (CVPR 2021). Uses box/segmentation annotations to edit causal vs non-causal regions; Waterbirds, IN-9, camera traps.
  [arXiv 2106.01127](https://arxiv.org/abs/2106.01127)
- **Ryali, Schwab, Morcos — Characterizing and Improving the Robustness of Self-Supervised Learning through Background
  Augmentations** (2021). Background swap in SSL with saliency masks; tolerant to substantial mask noise.
  [arXiv 2103.12719](https://arxiv.org/abs/2103.12719)
- **Tian et al. — Eliminating Background-bias for Robust Person Re-identification** (CVPR 2018). Random-background
  augmentation with person masks. [CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tian_Eliminating_Background-Bias_for_CVPR_2018_paper.pdf)
- **Aniraj, Dantas, Ienco, Marcos — Masking Strategies for Background Bias Removal in Computer Vision Models** (ICCV
  2023 Workshops). Early vs late masking, train on CUB, test on Waterbirds backgrounds; ConvNeXt and DINOv2 ViT.
  [arXiv 2308.12127](https://arxiv.org/abs/2308.12127)
- **Moayeri, Pope, Balaji, Feizi — A Comprehensive Study of Image Classification Model Sensitivity to Foregrounds,
  Backgrounds, and Visual Attributes** (CVPR 2022). RIVAL10 with masks; relative foreground sensitivity (RFS) via
  region noise; saliency alignment is similar across models despite large RFS differences.
  [arXiv 2201.10766](https://arxiv.org/abs/2201.10766)
- **Adebayo, Muelly, Abelson, Kim — Post hoc Explanations may be Ineffective for Detecting Unknown Spurious
  Correlation** (ICLR 2022). Attribution, concept and training-point methods fail when the artefact is unknown,
  worst for non-visible ones such as background blur. [arXiv 2212.04629](https://arxiv.org/abs/2212.04629)
- **Adebayo et al. — Sanity Checks for Saliency Maps** (NeurIPS 2018). (To cite; not re-read here.)
- **Rosenfeld, Zemel, Tsotsos — The Elephant in the Room** (arXiv 1808.03305, 2018). Transplanting an object into an
  unrelated scene changes detections non-locally — the detection analogue of "distraction".
  [arXiv](https://arxiv.org/abs/1808.03305)
- **Kirichenko, Izmailov, Wilson — Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations (DFR)**
  (ICLR 2023). Retrain the last layer on a small group-balanced set. Critique: Le et al. 2023, "Is Last Layer
  Re-Training Truly Sufficient...?" [arXiv 2204.02937](https://arxiv.org/abs/2204.02937),
  [arXiv 2308.00473](https://arxiv.org/abs/2308.00473)
- **Liu et al. — Just Train Twice (JTT)** (ICML 2021): closes ~75% of the ERM–Group DRO worst-group gap with group
  labels only on validation. [arXiv 2107.09044](https://arxiv.org/abs/2107.09044)
- **Waterbirds reference numbers** (ResNet-50, WGA, one consistent table, arXiv 2309.08534): class-balanced ERM 81.9,
  JTT 85.6, Group DRO-ES 90.7, DFR 91.1. ERM baselines differ a lot across papers (≈ 60–82); always quote one source.
- **"Seeing Through Distractions: Context-Stable Attribution via the Least Core"** (arXiv 2610.35827, 2026). Shapley
  and other attributions give large credit to contextual distractors even when the object suffices — relevant caveat
  for our region-Shapley numbers. [arXiv](https://arxiv.org/abs/2610.35827) (abstract-level only, to read)
- **Counterfactual Frequency (CoF) tables** (arXiv 2405.15661): background swaps to expose shortcuts that saliency
  maps miss. [arXiv](https://arxiv.org/abs/2405.15661) (abstract-level only)

Still to check: SAM-based variants of background augmentation 2023–2026, "context bias" work in object recognition
(e.g. Singh et al. 2020 "Don't judge an object by its context"), Spawrious, CounterAnimal, and whether any 2026 work
uses foreign-vs-own background attribution. ForAug and AutoBackSwap appendices were read only through summaries.
