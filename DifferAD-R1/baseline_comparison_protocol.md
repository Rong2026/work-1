# Baseline Comparison Protocol

This document summarizes the baseline evaluation protocol used for dual-image industrial anomaly localization. The goal is to make the comparison reproducible when different baselines have different native input assumptions, output formats, and task alignments.

In our setting, each sample contains a **reference image** and a **target image**. The model is asked to compare the two images, determine whether the target image contains a difference or anomaly, and localize the corresponding region in the **target image** using a single bounding box.

---

## 1. Evaluation Setting

### Input convention

| Field | Meaning |
|---|---|
| `Image-1` | Reference image, usually a defect-free or normal image |
| `Image-2` | Target image to be inspected |
| Prediction target | Bounding box of the different/anomalous region in `Image-2` |
| Normal case | Output a zero box, e.g., `(0,0),(0,0)` or an equivalent model-specific format |

### Unified semantic instruction

For all MLLM-based baselines that support multi-image input, we use the same semantic instruction:

```text
Please compare the two given images and determine whether there are any differences or anomaly.
If differences or anomaly exist, focus on identifying their approximate locations.
Output the localized different or anomaly regions using bounding boxes.
```

Only the **output-format constraint** is adapted to each model family, because different MLLMs use different native box representations.

---

## 2. MLLM Baselines

### Output format summary

| Baseline family | Model types | Required box format | Normal-image output |
|---|---|---|---|
| Qwen-VL series | `qwen_api`, `qwen_vl`, `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl` | `(x1,y1),(x2,y2)` | `(0,0),(0,0)` |
| UniVG-R1 | `univg-r1` | Final box in `<answer>...</answer>` using `(x1,y1),(x2,y2)` | `<answer>(0,0),(0,0)</answer>` |
| InternVL series | `internvl2_8b`, `internvl3`, `internvl3_5` | `[[x1,y1,x2,y2]]` | `[[0,0,0,0]]` |
| MiniCPM | `minicpm` | `x1 y1 x2 y2` | `0 0 0 0` |
| Mantis | `mantis` | `[x1, y1, x2, y2]` | `[0,0,0,0]` |
| mPLUG-Owl3 | `mplug_owl3` | `[x1, y1, x2, y2]` | `[0,0,0,0]` |
| CogVLM | `cogvlm` | `[[x0,y0,x1,y1]]` | `[[0,0,0,0]]` |
| MIGician | `migician` | Final box in `<answer>...</answer>` using `(x1,y1),(x2,y2)` | `<answer>(0,0),(0,0)</answer>` |

---

## 3. Full Prompts

### 3.1 Qwen-VL Series

Supported model types:

```text
qwen_api
qwen_vl
qwen2_vl
qwen2_5_vl
qwen3_vl
```

Prompt:

```text
Please compare the two given images and determine whether there are any differences or anomaly. If differences or anomaly exist, focus on identifying their approximate locations. Output the localized different or anomaly regions using bounding boxes.with coordinates formatted as (x1,y1),(x2,y2); if the images are exactly the same, output (0,0),(0,0). Format:<|box_start|>(x1,y1),(x2,y2)<|box_end|>. Don't generate addtional words.
```

Expected output:

```text
<|box_start|>(x1,y1),(x2,y2)<|box_end|>
```

---

### 3.2 UniVG-R1

Model type:

```text
univg-r1
```

Prompt:

```text
Please compare the two given images and determine whether there are any differences or anomaly. Provide a detailed description of the target image, and if differences or anomaly exist, focus on explaining the different regions and identify their approximate locations. Output the localized different or anomaly regions using bounding boxes, with coordinates formatted as (x1,y1),(x2,y2); if the images are exactly the same, output (0,0),(0,0).Enclose your reasoning process within <think></think> tags, and enclose your final bounding box answer within <answer></answer> tags.
```

Expected output:

```text
<think>reasoning process</think>
<answer>(x1,y1),(x2,y2)</answer>
```

During evaluation, only the final box inside the `<answer>` field is used.

---

### 3.3 InternVL Series

Supported model types:

```text
internvl2_8b
internvl3
internvl3_5
```

Image prefix:

```text
Image-1:<image>
Image-2:<image>
```

Prompt:

```text
Image-1:<image>
Image-2:<image>
Please compare the two given images and determine whether there are any differences or anomaly. If differences or anomaly exist, focus on identifying their approximate locations. Output the localized different or anomaly regions using bounding boxes.with coordinates formatted as [[x1,y1,x2,y2]]; if the images are exactly the same, output [[0,0,0,0]].Format:<box>[[x1,y1,x2,y2]]</box>. Don't generate addtional words.
```

Expected output:

```text
<box>[[x1,y1,x2,y2]]</box>
```

---

### 3.4 MiniCPM

Model type:

```text
minicpm
```

Prompt:

```text
Please compare the two given images and determine whether there are any differences or anomaly. If differences or anomaly exist, focus on identifying their approximate locations. Output the localized different or anomaly regions using bounding boxes.with coordinates formatted as x1 y1 x2 y2; if the images are exactly the same, output 0 0 0 0. Format:<box>x1 y1 x2 y2</box>. Don't generate addtional words.
```

Expected output:

```text
<box>x1 y1 x2 y2</box>
```

---

### 3.5 Mantis

Model type:

```text
mantis
```

Prompt:

```text
Please compare the two given images and determine whether there are any differences or anomaly. If differences or anomaly exist, focus on identifying their approximate locations. Output the localized different or anomaly regions using bounding boxes.with coordinates formatted as [x1, y1, x2, y2]; if the images are exactly the same, output [0,0,0,0]. Format:<box>[x1, y1, x2, y2]</box>. Don't generate addtional words.
```

Expected output:

```text
<box>[x1, y1, x2, y2]</box>
```

---

### 3.6 mPLUG-Owl3

Model type:

```text
mplug_owl3
```

Image prefix:

```text
Image-1:<|image|>
Image-2:<|image|>
```

Prompt:

```text
Image-1:<|image|>
Image-2:<|image|>
Please compare the two given images and determine whether there are any differences or anomaly. If differences or anomaly exist, focus on identifying their approximate locations. Output the localized different or anomaly regions using bounding boxes.with coordinates formatted as [x1, y1, x2, y2]; if the images are exactly the same, output [0,0,0,0]. Format:<box>[x1, y1, x2, y2]</box>. Don't generate addtional words.
```

Expected output:

```text
<box>[x1, y1, x2, y2]</box>
```

---

### 3.7 CogVLM

Model type:

```text
cogvlm
```

Prompt:

```text
Please compare the two given images and determine whether there are any differences or anomaly. If differences or anomaly exist, focus on identifying their approximate locations. Output the localized different or anomaly regions using bounding boxes.with coordinates formatted as [[x0,y0,x1,y1]]; if the images are exactly the same, output [[0,0,0,0]]. Coordinates format:[[x0,y0,x1,y1]].Ground it in the right image.
```

Expected output:

```text
[[x0,y0,x1,y1]]
```

The phrase `Ground it in the right image` is used to make the output box refer to the target image.

---

### 3.8 MIGician

Model type:

```text
migician
```

Prompt:

```text
Please compare the two given images and determine whether there are any differences or anomaly. Provide a detailed description of the target image, and if differences or anomaly exist, focus on explaining the different regions and identify their approximate locations. Output the localized different or anomaly regions using bounding boxes, with coordinates formatted as (x1,y1),(x2,y2); if the images are exactly the same, output (0,0),(0,0).Enclose your reasoning process within <think></think> tags, and enclose your final bounding box answer within <answer></answer> tags.
```

Expected output:

```text
<think>reasoning process</think>
<answer>(x1,y1),(x2,y2)</answer>
```

During evaluation, only the final box inside the `<answer>` field is used.

---

## 4. Output Parsing and Invalid Output Handling

All predicted boxes are parsed by the same evaluation script.

### Parsing rules

1. The parser first searches for the model-specific box pattern.
2. For reasoning-style models such as UniVG-R1 and MIGician, the final bounding box is extracted from the `<answer>` field.
3. If a model outputs multiple boxes, the first valid box is retained, following the single-anomaly setting used in the evaluation.
4. If no valid bounding box can be parsed, the prediction is treated as invalid.
5. Invalid predictions are assigned an IoU score of `0`.

### Normal samples

For samples without anomalies or differences, the expected prediction is a zero box. The exact zero-box representation depends on the model family:

```text
(0,0),(0,0)
[[0,0,0,0]]
0 0 0 0
[0,0,0,0]
```

---

## 5. Traditional Non-LLM Baselines

We also evaluate traditional anomaly detection baselines that are not originally designed for dual-image MLLM-style localization.

Included methods:

```text
PaDiM
PatchCore
WinCLIP
```

### Training and inference protocol

| Method | Training / reference protocol | Native output |
|---|---|---|
| PaDiM | Trained using only normal training images without defect labels | Anomaly map |
| PatchCore | Trained using only normal training images without defect labels | Anomaly map |
| WinCLIP | Uses normal reference images under the few-shot setting | Anomaly map |

These methods output anomaly maps rather than bounding boxes. Therefore, we convert anomaly maps into bounding boxes for localization evaluation.

### Anomaly-map-to-box conversion

1. Use 30% of the evaluation data as a calibration split.
2. Select the threshold that maximizes the F1-score on the calibration split.
3. Treat pixels above the calibrated threshold as anomalous pixels.
4. Extract connected components from the binary anomaly mask.
5. Select the largest connected component.
6. Convert the largest connected component into the predicted bounding box.
7. Use the converted box to compute localization metrics.

---

## 6. Metrics

We report localization metrics based on the predicted box and the ground-truth box.

| Metric | Description |
|---|---|
| `mIoU` | Mean Intersection-over-Union between predicted and ground-truth boxes |
| `Acc@0.3` | Accuracy under an IoU threshold of 0.3 |
| `F1-score` | Detection/localization F1-score under the selected protocol |

For invalid outputs, the IoU is set to `0`, which also affects downstream thresholded metrics.

---

## 7. Reproducibility Notes

The protocol is designed to avoid giving any method an advantage through inconsistent assumptions.

- All MLLM baselines receive the same reference-target image pair whenever the model supports multi-image input.
- The semantic task instruction is kept unchanged across MLLM baselines.
- Only the required output format is adapted to match each model's native convention.
- Traditional non-LLM baselines are evaluated using their standard anomaly-detection protocols.
- Since traditional baselines output anomaly maps, their predictions are converted into boxes using a fixed and reproducible thresholding-and-connected-component procedure.
- Invalid or unparsable MLLM outputs are not manually corrected; they are counted as invalid predictions with IoU `0`.
