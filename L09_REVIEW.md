# L09 Review (Round 5): Computer Vision — Seeing with Silicon

**Overall**: The lecture is ready to deliver. All issues from Rounds 1–4 are resolved. Content quality is high, jargon is defined at first use, every major section has reference cards + code snippets + visuals, prerequisite references are verified accurate, xkcd comics are well-distributed with no duplicates, and prior year content has been fully leveraged. One minor inconsistency remains.

---

## Detailed Findings

### 1. Ultralytics Table Inconsistent with Code (line 987)

The Specialized Libraries table says `YOLOv8+, batteries-included training` but the YOLO code snippet (line 745) now uses `YOLO("yolo11n.pt")`. These should match.

- **Suggestion**: Change table entry from `YOLOv8+, batteries-included training` to `YOLO11+, batteries-included training`.

---

## Verification Checks (All Pass)

### Prerequisite References ✓
- Line 143: "In lecture 06 you were introduced to CNNs using Keras" — **accurate**. L06 teaches Conv2D, MaxPooling2D with reference cards, Keras is primary framework, PyTorch only briefly previewed.
- Line 571: "The classification metrics from lecture 05 apply directly" — **accurate**. L05 covers accuracy, precision, recall, F1, AUROC, confusion matrix with formulas and code.
- Line 918: "self-attention" — covered thoroughly in L07 (transformers lecture, QKV mechanism, multi-head attention).
- PyTorch training loop taught from scratch in L09 — **correct**. L06 only showed a brief PyTorch preview; Keras was the primary framework.

### Jargon Definitions ✓
All key terms defined before or at first use:
- **Parameter explosion**, **parameter sharing**, **feature map** (line 147–149)
- **Backbone**, **head** (lines 475–476, dedicated bold definitions)
- **Skip connections** (line 503, parenthetical in ResNet table row)
- **Anchor boxes** (line 654, explained with example)
- **IoU** (line 658, formula given)
- **NMS** (line 662, explained with visual)
- **mAP** (line 666, explained with IoU thresholds)
- **FPN** (line 689, explained before detection table)
- **COCO** (line 701, defined inline)
- **Logits** (line 849, "raw unnormalized scores")
- **Pretext tasks** (line 951, examples given inline)
- **Depthwise separable convolutions** (line 504, inline explanation in architecture table)
- **Compound scaling** (line 505, inline explanation in architecture table)
- **Dilated convolutions** (line 856, inline explanation in torchvision segmentation table)
- **FCN** (line 857, inline explanation)

### xkcd Placement ✓ — 7 Comics, No Duplicates
| # | Comic | Line | Position |
|:--|:------|:-----|:---------|
| 1 | `xkcd_pixels.png` | 45 | Links → What Is CV |
| 2 | `file_extensions_2x.png` | 91 | Medical Formats → Python Stack |
| 3 | `xkcd_color_models.png` | 283 | Tensor Shapes → torchvision |
| 4 | `xkcd_predictive_models.png` | 561 | Transfer Learning → Evaluating |
| 5 | `xkcd_precision_vs_accuracy.png` | 632 | LIVE DEMO!! → Object Detection |
| 6 | `xkcd_ml_captcha.png` | 755 | Object Detection → Segmentation |
| 7 | `xkcd_heatmap.png` | 910 | Segmentation → Advanced Topics |

None appear in L01–L08. Verified against all 8 prior lectures (45 unique xkcd references across L01–L08, zero overlap).

Longest gap: 278 lines (xkcd 3 → xkcd 4), spanning torchvision + Transfer Learning. Both sections are image-dense with subject-matter visuals, so visual density is adequate.

### Code Correctness ✓
| Snippet | Check |
|:--------|:------|
| SimpleCNN `nn.Linear(64 * 56 * 56, 128)` | 224 / 2 / 2 = 56 after two MaxPool2d(2) ✓ |
| Training loop (zero_grad → forward → loss → backward → step) | Standard PyTorch pattern ✓ |
| Transfer learning: freeze → replace fc → optimize only fc.parameters() | Correct feature extraction workflow ✓ |
| `torchmetrics.ConfusionMatrix(task="binary")` | Valid torchmetrics API ✓ |
| `nn.CrossEntropyLoss(weight=torch.tensor([1.0, 9.0]).to(device))` | Device-aware ✓ |
| `YOLO("yolo11n.pt")` | Current Ultralytics YOLO11 ✓ |
| `smp.Unet(encoder_name="resnet18", ...)` / `smp.losses.DiceLoss(mode="binary")` | Valid smp API ✓ |
| MONAI `UNet(spatial_dims=2, ...)` | Valid MONAI API ✓ |
| `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)` | Valid timm API ✓ |

### Section Completeness ✓
| Section | Intro | Visual | Ref Card | Code |
|:--------|:-----:|:------:|:--------:|:----:|
| What Is CV / Images / Formats | ✓ | ✓ | ✓ | ✓ |
| CNNs in PyTorch | ✓ | ✓ | ✓ | ✓ |
| torchvision | ✓ | ✓ | ✓ | ✓ |
| Transfer Learning | ✓ | ✓ | ✓ | ✓ |
| Evaluating Vision Models | ✓ | FIXME | ✓ | ✓ |
| Object Detection | ✓ | ✓ | ✓ | ✓ |
| Image Segmentation | ✓ | ✓ | ✓ | ✓ |
| Advanced Topics | ✓ | ✓ | — | partial |

Advanced Topics is intentionally lighter ("concepts to be aware of, not necessarily to implement today" — line 914). All subsections now have at least one visual (ViT diagram, Grad-CAM example, SSL concept, robotic surgery CV, generative model concept).

### Prior Year Content ✓
Content from lectures_24/08 that has been incorporated:
- Parameter explosion calculation (150,528 × 1,000 = 150M+) ✓
- Filter/kernel explanation with parameter sharing ✓
- CNN architecture timeline (AlexNet → VGG → Inception → ResNet → EfficientNet → ConvNeXt) ✓
- Video analysis mention with image in Advanced Topics ✓
- Generative models (GANs, diffusion) mention with image in Advanced Topics ✓
- U-Net architecture with skip connections ✓
- Loss function comparison for segmentation ✓
- Self-supervised learning ✓

No significant content from lectures_24/08 remains unleveraged. Items intentionally omitted for scope: DICOM windowing/leveling, SimpleITK, optical flow — all niche topics that would add length without proportional pedagogical value.

### LIVE DEMO Markers ✓
Three markers at lines 453, 630, 1004.

### Media References ✓
- 39 images properly referenced (up from 37 in Round 4: added xkcd_heatmap.png, robotic_surgery_cv.png)
- 1 known missing (`nchw_vs_nhwc.png`, tagged with `#FIXME-missing-media`)
- 24 unused files in `media/` (cleanup opportunity, not a content issue)

---

## Summary of Suggestions (Priority Order)

1. **Update Ultralytics table entry** (line 987) — change "YOLOv8+" to "YOLO11+" to match code snippet
