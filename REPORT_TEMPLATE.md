# Medical Assistant LLM Fine-Tuning - Project Report Template

**Student Name:** Charlotte Kariza  
**Institution:** African Leadership University  
**Course:** Machine Learning Techniques  
**Date:** February 2026

---

## INSTRUCTIONS:
Replace all [FILL IN] sections with your own observations and experiences.
Answer each question in your own words based on what you actually did.

---

## 1. ABSTRACT (Write this LAST - 150-200 words)

[FILL IN: In 3-4 sentences, summarize:]
- What problem you addressed
- What you did (fine-tuning method)
- Key results (BLEU score improvement)
- Main conclusion

---

## 2. INTRODUCTION

### 2.1 Motivation
**Question to answer:** Why did you choose this project? What real-world problem does it solve?

[FILL IN: Write 2-3 paragraphs about:]
- Why medical information accessibility matters to you
- Who would benefit from this tool
- What gap you're trying to fill

### 2.2 Project Objectives
List what you aimed to achieve:

1. [FILL IN: e.g., "Fine-tune a language model on medical Q&A data"]
2. [FILL IN: e.g., "Compare different hyperparameter configurations"]
3. [FILL IN: e.g., "Deploy a working chatbot interface"]

---

## 3. BACKGROUND & RELATED WORK

### 3.1 Large Language Models
**Questions to answer:**
- What is TinyLlama and why did you choose it?
- What are the limitations of large models?

[FILL IN: Explain in your own words what you learned about LLMs]

### 3.2 Parameter-Efficient Fine-Tuning (PEFT)
**Questions to answer:**
- What is LoRA? How does it work?
- Why use LoRA instead of full fine-tuning?

[FILL IN: Describe LoRA based on what you understand from your training]

### 3.3 Medical AI Applications
**Questions to answer:**
- What other medical AI projects have you seen?
- How is yours different?

[FILL IN: Mention 2-3 similar projects or papers you referenced]

---

## 4. METHODOLOGY

### 4.1 Dataset

**Original Source:** MedQA Medical Flashcards (MedAlpaca collection)
**Total samples:** 33,000+
**Samples used:** [FILL IN: How many did you use?]
**Split:** [FILL IN: How did you split train/validation?]

**Data Preprocessing Steps:**
[FILL IN: List the steps you took. For example:]
1. Loaded dataset using...
2. Filtered out...
3. Formatted into...
4. Split into...

**Question:** Why did you choose this dataset size? What trade-offs did you consider?
[FILL IN: Your reasoning]

### 4.2 Base Model Selection

**Model:** TinyLlama-1.1B-Chat-v1.0

**Why this model?**
[FILL IN: Your reasons - consider:]
- Size constraints
- Available compute
- Pre-training quality
- Community support

### 4.3 Fine-Tuning Configuration

**Method:** LoRA (Low-Rank Adaptation)

**Hyperparameters you tested:**

| Parameter | Value(s) Tested | Why? |
|-----------|----------------|------|
| Learning Rate | [FILL IN] | [FILL IN: Why these values?] |
| Batch Size | [FILL IN] | [FILL IN: Memory constraints?] |
| LoRA Rank | [FILL IN] | [FILL IN: Model capacity?] |
| Epochs | [FILL IN] | [FILL IN: Overfitting concerns?] |

**Training Environment:**
- Platform: [FILL IN: Google Colab? Local?]
- GPU: [FILL IN: T4? A100?]
- Time per experiment: [FILL IN: approximately how long?]

---

## 5. EXPERIMENTS

### 5.1 Experiment Design

**Research Question:** [FILL IN: What were you trying to find out?]

**Experiments Conducted:**

#### Experiment 1: Baseline
- Configuration: [FILL IN: learning rate, batch size, etc.]
- Rationale: [FILL IN: Why start here?]
- Final Loss: [FILL IN]
- Observations: [FILL IN: What did you notice during training?]

#### Experiment 2: [FILL IN: Name it]
- Configuration: [FILL IN]
- Rationale: [FILL IN: What were you testing?]
- Final Loss: [FILL IN]
- Observations: [FILL IN]

[REPEAT for experiments 3, 4, 5]

### 5.2 What Went Wrong? (Important!)

**Question:** Did any experiments fail? What problems did you encounter?

[FILL IN: Be honest about:]
- Training crashes
- Out of memory errors
- Poor results
- Unexpected behaviors

**How did you fix these issues?**
[FILL IN: Your debugging process]

---

## 6. RESULTS

### 6.1 Training Metrics

**Best Model:** [FILL IN: Which experiment?]

**Comparison Table:**

| Metric | Base Model | Your Model | Improvement |
|--------|------------|------------|-------------|
| BLEU Score | [FILL IN] | [FILL IN] | [FILL IN]% |
| ROUGE-1 | [FILL IN] | [FILL IN] | [FILL IN]% |
| ROUGE-2 | [FILL IN] | [FILL IN] | [FILL IN]% |
| Training Loss | N/A | [FILL IN] | N/A |

**Question:** Were you surprised by any results? Why?
[FILL IN: Your analysis]

### 6.2 Qualitative Analysis

**Test Questions You Used:**
[FILL IN: List 5-7 medical questions you tested]

**Example 1:**
- **Question:** [FILL IN]
- **Base model response:** [FILL IN]
- **Your model response:** [FILL IN]
- **Your assessment:** [FILL IN: Better? Worse? Why?]

[REPEAT for 3-5 examples]

### 6.3 Where Your Model Struggles

**Question:** What types of questions does your model fail on?

[FILL IN: Identify patterns:]
- Complex medical calculations?
- Rare diseases?
- Drug dosages?
- Emergency situations?

**Give specific examples of failures:**
[FILL IN: 2-3 bad outputs and why they're problematic]

---

## 7. DEPLOYMENT

### 7.1 Gradio Interface

**Why Gradio?**
[FILL IN: Why did you choose this framework?]

**Features you implemented:**
- [FILL IN: Chat history?]
- [FILL IN: Temperature control?]
- [FILL IN: Medical disclaimer?]

### 7.2 Deployment Challenges

**Question:** What problems did you face deploying to Hugging Face Spaces?

[FILL IN: Document the actual issues:]
- Python version problems?
- Package conflicts?
- Model loading errors?
- API issues?

**How you solved each problem:**
[FILL IN: Be specific about your solutions]

### 7.3 Final Demo

**Live URL:** [FILL IN: Your Hugging Face Space link]
**GitHub:** https://github.com/karizacharlotte/Domain-Specific-Assistant-via-LLMs-Fine-Tuning

---

## 8. DISCUSSION

### 8.1 Key Learnings

**Technical Learnings:**
[FILL IN: What did you learn about:]
- Fine-tuning LLMs?
- LoRA technique?
- Hyperparameter tuning?
- Model evaluation?

**Practical Learnings:**
[FILL IN: What did you learn about:]
- GPU resource management?
- Debugging ML models?
- Deploying AI applications?
- Working with medical data?

### 8.2 Limitations

**Be critical of your own work:**

[FILL IN: Honestly assess:]
1. Model limitations (what can't it do?)
2. Data limitations (biases, coverage gaps?)
3. Evaluation limitations (are BLEU scores enough?)
4. Ethical concerns (medical advice risks?)
5. Resource limitations (how would more compute help?)

### 8.3 Future Improvements

**If you had more time/resources:**

[FILL IN: List 5-8 specific improvements:]
1. [e.g., "Expand training data to include...]
2. [e.g., "Implement retrieval-augmented generation for...]
3. [etc.]

---

## 9. CONCLUSION

[FILL IN: Write 2-3 paragraphs summarizing:]
- What you accomplished
- Whether you met your objectives
- Most important takeaway
- Broader impact of this work

---

## 10. REFERENCES

[FILL IN: List all resources you used:]

**Papers:**
1. [Full citation]
2. [Full citation]

**Datasets:**
1. [Full citation]

**Tools & Libraries:**
1. [Full citation]

**Other Resources:**
1. [Any tutorials, blog posts, documentation]

---

## APPENDIX

### A. Code Snippets
[FILL IN: Include 2-3 key code snippets with explanations]

### B. Additional Results
[FILL IN: Extra graphs, tables, examples]

### C. Deployment Documentation
[FILL IN: Screenshots of your running app]

---

## SELF-EVALUATION CHECKLIST

Before submitting, check:
- [ ] All [FILL IN] sections completed
- [ ] No generic AI-generated sentences
- [ ] All results are YOUR actual results
- [ ] All opinions/observations are YOUR thoughts
- [ ] Proper citations for all references
- [ ] Honest about failures and limitations
- [ ] Technical accuracy verified
- [ ] Grammar and spelling checked
- [ ] Figures/tables have captions
- [ ] Code is properly formatted
- [ ] Page numbers added
- [ ] Table of contents created

---

**WORD TARGET:** 3000-5000 words (adjust based on your course requirements)

**WRITING TIPS:**
1. Use first person ("I chose...", "I observed...")
2. Be specific (use actual numbers from YOUR experiments)
3. Be honest (admit what didn't work)
4. Explain your reasoning (why did you make each choice?)
5. Connect to course concepts (reference what you learned in class)
