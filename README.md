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

### AD-DualDiff Dataset
<div align="center">
  <a href="AD-DualDiff/dataset_pairs_show.pdf">
    <img src="https://img.shields.io/badge/View-Dataset%20Showcase-4CAF50?style=for-the-badge&logo=pdf&logoColor=white" alt="Dataset Pairs Showcase">
  </a>
  <p><em>AD-DualDiff dataset showcasing paired normal/anomalous images across 20 industrial categories</em></p>
</div>

The AD-DualDiff dataset comprises 13K carefully curated image pairs spanning 20 industrial categories, designed specifically for difference-based anomaly localization tasks.

### Pipeline Overview
<div align="center">
  <a href="models/pipeline.pdf">
    <img src="https://img.shields.io/badge/View-Training%20Pipeline-2196F3?style=for-the-badge&logo=pdf&logoColor=white" alt="Training Pipeline">
  </a>
  <p><em>DifferAD-R1 training pipeline with dual-image paradigm and reinforcement learning</em></p>
</div>

## 💬 Interactive Examples

### Model Dialogues
<div align="center">
  <a href="output_show/duihua.pdf">
    <img src="https://img.shields.io/badge/View-Model%20Dialogues-FF9800?style=for-the-badge&logo=pdf&logoColor=white" alt="Model Dialogues">
  </a>
  <p><em>Interactive dialogues showing anomaly localization reasoning process</em></p>
</div>

### Visualization Results
<div align="center">
  <a href="output_show/display.pdf">
    <img src="https://img.shields.io/badge/View-Localization%20Results-9C27B0?style=for-the-badge&logo=pdf&logoColor=white" alt="Localization Results">
  </a>
  <p><em>Visualization of anomaly localization results across different industrial categories</em></p>
</div>