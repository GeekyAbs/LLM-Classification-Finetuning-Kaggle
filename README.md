<div align="center">

# 🏆 Top 30 Solution: LLM Classification Finetuning (Kaggle)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg?style=for-the-badge&logo=huggingface)](https://huggingface.co/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20beff.svg?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/llm-classification-finetuning)


</div>

---

## 📖 Overview

This repository contains our team's complete codebase, experiment logs, and progression for the **[Kaggle LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)** competition. 

The objective is to predict human preference between two Large Language Models based on a user's prompt and the models' respective responses. Given a prompt and two responses (Model A and Model B), the system predicts the probability distribution across three outcomes: `[Model A Wins, Model B Wins, Tie]`.

> **🏆 Result:** Out of 210+ participating teams, our final ensembled solution achieved a **Top 30** position on the leaderboard.

---

## 🚀 Methodology & Progression

We treated this competition as a progressive ladder, building up architectural complexity only when our intuition and validation results demanded it.

### 1. The Humble Baseline
We believe in establishing a rock-solid, explainable baseline before burning GPU hours. 
* **Approach:** Extracted TF-IDF features from the prompts and responses, fed into a Logistic Regression classifier.
* **Outcome:** Established our baseline Log Loss score. 

### 2. Scaling to Encoder Models
Once our baseline was set, we moved to the traditional heavy-hitters for NLP text classification.
* **Approach:** Fine-tuned Masked Language Models (MLMs), specifically **DeBERTa** and **ModernBERT**. 
* **Outcome:** Stalled at the **~190th position**. The models struggled to grasp the conversational context and subjective nuance required to judge "helpfulness."

### 3. The Pivot to Causal LLMs
*Insight: If the task requires judging an LLM, why not use an LLM?*
* **Approach:** Applied Low-Rank Adaptation (LoRA) to fine-tune decoder-only models (**Llama-3.2-1B** and **Qwen-2.5-0.5B**).
* **Outcome:** Improved to the **~140th position**. However, the models were bottlenecked by the massive sequence lengths of multi-turn conversations, leading to context drowning.

---

## 💡 Key Breakthrough: Smart Truncation

Instead of throwing more compute at the problem to artificially increase context windows, we re-evaluated our data pipeline. 

In a chatbot response, the most critical markers of "helpfulness" are usually found in the **introduction** *(Did it directly answer the prompt?)* and the **conclusion** *(Did it summarize or format the code correctly?)*. The middle is often repetitive noise.

We implemented a **Smart Truncation** preprocessing step:
1. Retain the first $N$ characters.
2. Retain the last $N$ characters.
3. Completely strip the noisy middle segment.

> **🚀 Impact:** This single preprocessing optimization was our magic bullet, catapulting our single Qwen-0.5B model directly into the **Top 30**.

### The Final Polish: Ensembling
To stabilize our Log Loss score against the hidden test set, we ensembled our best models to capture diverse reasoning capabilities.
* **Method:** Weighted mean ensemble.
* **Result:** Highly stable, robust predictions that minimized overconfidence penalties.

---