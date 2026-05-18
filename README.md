# DifferAD-R1: A Difference-Guided Industrial Anomaly Localization with Multimodal Large Language Models


Industrial anomaly localization aims to accurately identify and localize abnormal regions in industrial products, addressing the critical challenge of detecting unseen defect categories in real-world scenarios. Traditional closed-set methods often suffer from poor cross-scenario generalization, while existing Multimodal Large Language Model (MLLM)-based approaches face two core limitations: they either adopt QA-style paradigms misaligned with the practical demands of localization, or rely on standard optimization techniques such as Group Relative Policy Optimization (GRPO), which fails to deliver effective learning signals for subtle defects. To tackle these issues, this paper proposes DifferAD-R1, an MLLM-augmented reinforcement learning framework tailored for industrial anomaly localization. We design a Difference-Guided dual-image paradigm, which reformulates the localization task as a one-shot difference grounding problem to effectively explore cross-scenario anomalies. A Dual-Consistency Localization Reward is developed for hard-to-detect anomalies, enhancing optimization stability and robustness. Additionally, we integrate a difficulty-aware strategy with adaptive reweighting and group-wise resampling to prioritize learning on challenging instances. To facilitate evaluations in real-world industrial settings, we construct the AD-DualDiff dataset, comprising 13K paired images across 20 categories. Experimental results demonstrate that DifferAD-R1 significantly outperforms existing baselines and achieves competitive performance compared to large-scale models like Qwen3-VL (235B parameters).

## 🎯 Key Features

- **Difference-Guided Dual-Image Paradigm**: Reformulates anomaly localization as a one-shot difference grounding problem
- **Dual-Consistency Localization Reward**: Enhanced optimization stability for hard-to-detect anomalies
- **Difficulty-Aware Strategy**: Adaptive reweighting and group-wise resampling for challenging instances
- **AD-DualDiff Dataset**: 13K paired images across 20 industrial categories
- **Superior Performance**: Outperforms existing baselines and competes with large-scale models like Qwen3-VL (235B parameters)

## 📊 Method Overview

Traditional closed-set methods suffer from poor cross-scenario generalization, while existing MLLM-based approaches either use misaligned QA-style paradigms or rely on standard GRPO that fails to provide effective learning signals for subtle defects.

DifferAD-R1 addresses these limitations through:
- **Dual-Image Input**: Simultaneous processing of normal and anomalous images
- **Reinforcement Learning**: GRPO-based optimization with specialized rewards
- **Adaptive Sampling**: Difficulty-aware resampling for better training efficiency

## 📈 Experimental Results

DifferAD-R1 demonstrates significant improvements over existing baselines across multiple industrial anomaly detection benchmarks, achieving competitive performance compared to models with substantially larger parameter counts.

## 📁 Dataset & Resources

## 📦 AD-DualDiff Dataset

<p align="center">
  <a href="https://github.com/Rong2026/work-1/blob/main/DifferAD-R1/AD-DualDiff/dataset_pairs_show.png">
    <img src="https://raw.githubusercontent.com/Rong2026/work-1/main/DifferAD-R1/AD-DualDiff/dataset_pairs_show.png" alt="AD-DualDiff Dataset Showcase" width="95%" />
  </a>
</p>

<p align="center">
  <em>
    AD-DualDiff dataset showcasing paired normal/anomalous images across 20 industrial categories
  </em>
</p>

The **AD-DualDiff dataset** comprises **13K carefully curated paired images** spanning **20 industrial categories**, specifically designed for **difference-based industrial anomaly localization** under one-shot and open-set settings.

---

## 🔧 Pipeline Overview

<p align="center">
  <a href="https://github.com/Rong2026/work-1/blob/main/DifferAD-R1/models/pipeline.png">
    <img src="https://raw.githubusercontent.com/Rong2026/work-1/main/DifferAD-R1/models/pipeline.png" alt="DifferAD-R1 Training Pipeline" width="95%" />
  </a>
</p>

<p align="center">
  <em>
    DifferAD-R1 training pipeline with a dual-image paradigm and reinforcement learning
  </em>
</p>

---

## 💬 Interactive Examples

### 🧠 Model Dialogues

<p align="center">
  <a href="https://github.com/Rong2026/work-1/blob/main/DifferAD-R1/output_show/duihua.png">
    <img src="https://raw.githubusercontent.com/Rong2026/work-1/main/DifferAD-R1/output_show/duihua.png" alt="Model Dialogues" width="95%" />
  </a>
</p>

<p align="center">
  <em>
    Interactive dialogues illustrating the anomaly localization reasoning process
  </em>
</p>

---

### 🎯 Visualization Results

<p align="center">
  <a href="https://github.com/Rong2026/work-1/blob/main/DifferAD-R1/output_show/display.png">
    <img src="https://raw.githubusercontent.com/Rong2026/work-1/main/DifferAD-R1/output_show/display.png" alt="Anomaly Localization Results" width="95%" />
  </a>
</p>

<p align="center">
  <em>
    Visualization of anomaly localization results across diverse industrial categories
  </em>
</p>
---

### 🎬 Industrial Deployment Demo on a 2B Model

<p align="center">
  <video src="https://github.com/Rong2026/work-1/raw/main/DifferAD-R1/output_show/industrial_deployment_2B.mp4" controls width="95%">
    Your browser does not support the video tag.
  </video>
</p>

<p align="center">
  <a href="https://github.com/Rong2026/work-1/blob/main/DifferAD-R1/output_show/industrial_deployment_2B.mp4">
    ▶ Watch the 2B Industrial Deployment Demo
  </a>
</p>

<p align="center">
  <em>
    Industrial deployment demo showing the transfer of DifferAD-R1 to a lightweight 2B-scale model for real-world anomaly localization.
  </em>
</p>

