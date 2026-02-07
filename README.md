# EAA: Emotion-Aware Audio Large Language Models  
## Dual Cross-Attention with Context-Aware Instruction Tuning

Repository for the Interspeech 2025 paper:

> **EAA: Emotion-Aware Audio Large Language Models with Dual Cross-Attention and Context-Aware Instruction Tuning**  
> Hongfei Du, Sidi Lu, Gang Zhou, Ye Gao  
> Department of Computer Science, William & Mary  
> Interspeech 2025, Rotterdam, The Netherlands  

---

## Overview

Understanding emotion in speech is crucial for human-computer interaction and mental health monitoring. While audio large language models (ALLMs) have demonstrated strong performance in speech understanding, they often struggle to:

- Effectively integrate acoustic and semantic features  
- Capture dialogue-level emotional context  
- Dynamically adjust feature importance based on emotional cues  

To address these challenges, we propose **EAA**, an emotion-aware audio LLM framework that introduces:

- **Dual Cross-Attention Mechanism** for adaptive fusion of acoustic and semantic representations  
- **Context-Aware Instruction Tuning** to incorporate dialogue history into emotion recognition  

On the MELD dataset, EAA significantly outperforms existing ALLMs, achieving a **+11.4% improvement in accuracy**.