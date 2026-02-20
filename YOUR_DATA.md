# YOUR PROJECT DATA - Quick Reference

Use this to fill in the template with YOUR actual work.

---

## YOUR ACTUAL EXPERIMENTS (from README.md):

| Experiment | Learning Rate | Batch Size | LoRA Rank | Epochs | Final Loss |
|------------|---------------|------------|-----------|--------|------------|
| Baseline   | 2e-4         | 4          | 16        | 1      | 1.2463     |
| Lower LR   | 1e-4         | 4          | 16        | 1      | 1.2347     |
| Higher LoRA| 2e-4         | 4          | 32        | 1      | 1.1856     |
| Smaller Batch| 2e-4       | 2          | 16        | 1      | 1.2186     |
| More Epochs| 1.5e-4       | 4          | 16        | 2      | **1.0823** |

**Best Model:** More Epochs (13.1% improvement)

---

## YOUR ACTUAL RESULTS:

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| BLEU Score | 12.34 | 28.67 | +132% |
| ROUGE-1 | 0.21 | 0.48 | +125% |
| ROUGE-2 | 0.09 | 0.22 | +151% |

---

## YOUR TECHNICAL STACK:

- **Base Model:** TinyLlama-1.1B-Chat-v1.0
- **Dataset:** MedQA (MedAlpaca) - used 5,000 samples
- **Split:** 4,500 training, 500 validation
- **Method:** LoRA (Low-Rank Adaptation)
- **Platform:** Google Colab with T4 GPU
- **Framework:** Hugging Face Transformers, PEFT, TRL
- **Interface:** Gradio
- **Deployment:** Hugging Face Spaces with Docker

---

## YOUR DEPLOYMENT ISSUES (that you actually faced):

1. **Python 3.13 audioop error** - Fixed with Docker + Python 3.10
2. **@gr.load decorator error** - Removed invalid decorator
3. **HfFolder import error** - Pinned Gradio 4.29.0 + huggingface-hub 0.20.3
4. **GPU quantization on CPU** - Added CPU/GPU auto-detection
5. **API schema parsing error** - Downgraded Gradio version
6. **Configuration error** - Added YAML frontmatter to README

---

## EXAMPLE Q&A (from your README):

**Question:** "What is hypertension and how is it treated?"

**Base Model:** *Vague response lacking medical specificity*

**Your Fine-Tuned Model:** "Hypertension, or high blood pressure, is a condition where blood pressure in the arteries is consistently elevated (≥140/90 mmHg). Treatment includes: 1) Lifestyle modifications: reduced sodium intake, regular exercise, weight management, stress reduction. 2) Medications: ACE inhibitors, ARBs, beta-blockers, calcium channel blockers, or diuretics, depending on severity and patient factors. Regular monitoring is essential."

---

## QUESTIONS TO TEST (use these in your report):

1. "What is hypertension and how is it treated?"
2. "Explain the difference between bacteria and viruses."
3. "What are the symptoms of diabetes?"
4. "How does the immune system work?"
5. "What is the purpose of vaccination?"

---

## YOUR LORA CONFIGURATION:

```python
lora_config = LoraConfig(
    r=16,              # LoRA rank (or 32 for higher LoRA experiment)
    lora_alpha=32,     # Alpha scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## TRAINING PARAMETERS:

```python
training_args = TrainingArguments(
    num_train_epochs=1 or 2,
    per_device_train_batch_size=4 or 2,
    learning_rate=2e-4, 1e-4, or 1.5e-4,
    fp16=True,
    logging_steps=10,
    optim="paged_adamw_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)
```

---

## YOUR PROJECT FILES:

1. Medical_Assistant_LLM_FineTuning.ipynb - Main training notebook
2. app.py - Gradio chatbot (249 lines)
3. Dockerfile - Python 3.10 deployment config
4. requirements.txt - All dependencies
5. models/final/ - Best trained model (8.7 MB)
6. README.md - Documentation

---

## DEPLOYMENT URLS:

- **GitHub:** https://github.com/karizacharlotte/Domain-Specific-Assistant-via-LLMs-Fine-Tuning
- **Hugging Face Space:** [Add your space URL here]
- **Colab Notebook:** https://colab.research.google.com/drive/19vCx0wqRTY5a2hM1hOIEsDlBnCzH6z2f

---

## HOW TO USE THIS:

1. Open REPORT_TEMPLATE.md
2. For each [FILL IN], look here for your actual data
3. Write in YOUR OWN WORDS explaining what you did
4. Include these numbers/facts but explain them yourself
5. Add your personal observations and learnings

---

## IMPORTANT NOTES FOR WRITING:

### What to emphasize:
- You chose TinyLlama because it fits in Colab's free T4 GPU
- You ran 5 experiments to systematically test hyperparameters
- More epochs (2) worked better than higher rank or lower learning rate
- Deployment was challenging but you debugged systematically
- BLEU scores improved 132% but still have limitations

### What to be honest about:
- Free Colab means limited compute (can't use 7B models)
- Only used 5,000 samples (not full 33K dataset)
- BLEU/ROUGE don't fully capture medical accuracy
- Model runs on CPU in HF Spaces (slow inference)
- Can't handle dosage calculations or emergency advice
- Needs more diverse medical topics

### What made your project unique:
- Systematic hyperparameter experimentation (5 configs)
- Full deployment pipeline (training → evaluation → deployment)
- Overcame multiple deployment challenges
- Added medical disclaimers for safety
- CPU/GPU auto-detection for flexibility
