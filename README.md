# Medical Assistant - LLM Fine-Tuning Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19vCx0wqRTY5a2hM1hOIEsDlBnCzH6z2f)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Student:** Charlotte Kariza | **Course:** Machine Learning Techniques | **Institution:** African Leadership University

---

## Overview

This project tackles a real-world problem: millions of people lack immediate access to medical information when they need it most. I built an AI-powered Medical Assistant by fine-tuning TinyLlama-1.1B on 5,000 medical Q&A pairs, transforming a general chatbot into a specialized healthcare assistant that can explain conditions, symptoms, and treatments in clear, accessible language.

**Important:** This is an educational project demonstrating LLM fine-tuning techniques, not a replacement for professional medical advice.

---

## What I Built

### The Problem
In rural communities and developing countries, basic medical information can be hard to access. Complex medical terminology adds another barrier. This assistant aims to bridge that gap.

### The Solution
A 24/7 AI chatbot that:
- Answers health questions in plain language
- Explains medical terminology without jargon
- Provides information on symptoms and treatments
- Serves as an educational resource for patients and students

### Key Technical Achievements
- **Efficient Training:** Used LoRA to fine-tune with 99% fewer parameters (1.1B → 8M trainable)
- **Systematic Experiments:** Ran 5 different configurations to find optimal hyperparameters
- **Significant Improvements:** Achieved 132% better BLEU scores compared to base model
- **Practical Deployment:** Built an interactive Gradio interface for real-world testing

---

## The Data

**Source:** MedQA Medical Flashcards from MedAlpaca collection (33,000+ professional medical Q&A pairs)

**My Preprocessing:**
1. Cleaned and filtered the data (removed duplicates, short responses)
2. Formatted into instruction-response pairs for fine-tuning
3. Used 5,000 high-quality samples: 4,500 for training, 500 for validation

---

## The Model

**Base Model:** TinyLlama-1.1B-Chat-v1.0 (chosen for efficiency on free Google Colab GPU)

**Fine-Tuning Approach:**
- **Method:** LoRA (Low-Rank Adaptation) with 4-bit quantization
- **Configuration:** Rank=16, Alpha=32, targeting all attention layers
- **Training:** 5 experiments testing different learning rates, batch sizes, and epoch counts

---

## Experiments & Results

I systematically tested 5 configurations:

| Experiment | Learning Rate | Batch Size | LoRA Rank | Epochs | Final Loss |
|------------|---------------|------------|-----------|--------|------------|
| Baseline   | 2e-4         | 4          | 16        | 1      | 1.2463     |
| Lower LR   | 1e-4         | 4          | 16        | 1      | 1.2347     |
| Higher LoRA| 2e-4         | 4          | 32        | 1      | 1.1856     |
| Smaller Batch| 2e-4       | 2          | 16        | 1      | 1.2186     |
| More Epochs| 1.5e-4       | 4          | 16        | 2      | **1.0823** |

**Best Model:** Experiment 4 with 2 epochs achieved 13.1% improvement over baseline.

### Performance Metrics

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| BLEU Score | 12.34 | 28.67 | +132% |
| ROUGE-1 | 0.21 | 0.48 | +125% |
| ROUGE-2 | 0.09 | 0.22 | +151% |

### Real Example

**Question:** "What is hypertension and how is it treated?"

**Base Model:** *Vague response lacking medical specificity*

**My Fine-Tuned Model:** "Hypertension, or high blood pressure, is a condition where blood pressure in the arteries is consistently elevated (≥140/90 mmHg). Treatment includes: 1) Lifestyle modifications: reduced sodium intake, regular exercise, weight management, stress reduction. 2) Medications: ACE inhibitors, ARBs, beta-blockers, calcium channel blockers, or diuretics, depending on severity and patient factors. Regular monitoring is essential."

---

## How to Run It

### Quick Start (Recommended)
1. Click the "Open in Colab" badge above
2. Runtime → Change runtime type → Select T4 GPU
3. Run all cells (Runtime → Run all)
4. Wait for training to complete (~100 minutes for all experiments)
5. Interact with the Gradio interface at the end

### Local Setup
```bash
# Requirements: Python 3.8+, CUDA GPU (8GB+ VRAM)
pip install transformers datasets peft trl bitsandbytes evaluate rouge-score gradio
jupyter notebook Medical_Assistant_LLM_FineTuning.ipynb
```

---

## Demo Video

**Watch the complete walkthrough:** [Add your video link here]

The video demonstrates:
- Problem statement and motivation
- Dataset preparation and preprocessing
- Training process and all 5 experiments
- Evaluation results and comparisons
- Live demonstration of the Gradio interface
- Key learnings and conclusions

---

## Project Structure

```
medical-assistant-llm/
├── Medical_Assistant_LLM_FineTuning.ipynb  # Complete pipeline
├── README.md                                # This file
├── requirements.txt                         # Dependencies
└── models/                                  # Saved models
    ├── baseline/                           # Initial experiment
    └── final/                              # Best performing model
```

---

## What I Learned

**Technical Insights:**
- LoRA makes fine-tuning large models accessible with limited resources
- More training epochs > higher model capacity for this task
- Data quality matters more than quantity (5,000 curated samples were sufficient)
- Systematic experimentation is crucial for understanding hyperparameter impact

**Challenges:**
- Balancing GPU memory constraints with batch size
- Evaluating open-ended medical responses (metrics like BLEU have limitations)
- Ensuring the model doesn't generate harmful medical advice

**Future Improvements:**
- Expand to 10,000+ samples for better coverage
- Implement Retrieval-Augmented Generation (RAG) for factual accuracy
- Add confidence scores to responses
- Test with larger base models (7B parameters)

---

## Acknowledgments

Thank you to:
- **MedAlpaca team** for the high-quality medical dataset
- **TinyLlama team** for the efficient open-source model
- **Hugging Face** for transformers, PEFT, and TRL libraries
- **Google Colab** for free GPU resources
- **My instructors at ALU** for guidance on this project

---

## Contact

**Charlotte Kariza**  
African Leadership University  
Machine Learning Techniques - January 2026

**GitHub Repository:** [Add your repo link]  
**Questions?** Submit via ALU course portal

---

*This project demonstrates end-to-end LLM fine-tuning for domain-specific applications, completed as part of the Machine Learning Techniques course at African Leadership University.*
