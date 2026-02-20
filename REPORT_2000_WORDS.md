# Medical Assistant: Fine-Tuning TinyLlama for Healthcare Information Access

**Charlotte Kariza**  
African Leadership University  
Machine Learning Techniques  
February 2026

---

## Abstract

Access to medical information remains a challenge in many underserved communities worldwide. This project addresses this gap by fine-tuning TinyLlama-1.1B-Chat on 5,000 medical question-answer pairs using LoRA (Low-Rank Adaptation). I systematically tested five training configurations to optimize performance within free GPU constraints. The best model, trained with 2 epochs and a learning rate of 1.5e-4, achieved a BLEU score of 28.67—a 132% improvement over the base model—using only 8 million trainable parameters. I deployed the model as an interactive chatbot on Hugging Face Spaces, overcoming several technical challenges including Python compatibility issues and CPU deployment constraints. This demonstrates that effective medical AI assistants can be built with limited resources through careful hyperparameter tuning and efficient fine-tuning methods.

---

## 1. Introduction

During my time at ALU, I became increasingly aware of healthcare access disparities, particularly in communities where medical information isn't readily available or is difficult to understand. Large language models have shown impressive capabilities, but general-purpose models often lack specialized medical knowledge. I wanted to explore whether I could take a small, efficient language model and specialize it for medical question-answering—creating an educational tool that could explain medical concepts in clear language.

My objectives were threefold: (1) fine-tune a language model on medical Q&A data using methods compatible with free GPU resources, (2) systematically experiment with different hyperparameters to understand what works best, and (3) deploy a working chatbot interface that people could actually use. Throughout the project, I also wanted to develop practical intuition for modern fine-tuning techniques and gain experience with the complete ML pipeline from data preprocessing through deployment.

---

## 2. Methodology

### 2.1 Model and Dataset Selection

I chose TinyLlama-1.1B-Chat-v1.0 as my base model for practical reasons. While larger models like Llama-2-7B might perform better, they require around 28GB of GPU memory for fine-tuning—far exceeding the 16GB available on Google Colab's free T4 GPU. TinyLlama's 1.1 billion parameters offered a sweet spot: small enough to fine-tune on free resources but large enough to learn meaningful patterns.

For data, I used the MedQA Medical Flashcards dataset from the MedAlpaca collection, which contains over 33,000 professionally-written medical Q&A pairs. Given computational constraints, I used 5,000 samples split into 4,500 for training and 500 for validation. I preprocessed the data by removing duplicates, filtering very short answers, and reformatting into TinyLlama's chat template with system prompts, user questions, and assistant responses.

### 2.2 Fine-Tuning Approach

I used LoRA (Low-Rank Adaptation), which adds small "adapter" matrices to specific layers rather than updating all model weights. With a LoRA rank of 16, I trained only about 8 million parameters—less than 1% of the model's total size. The adapters targeted the attention layers (q_proj, k_proj, v_proj, o_proj) where most learning happens.

I used 4-bit NF4 quantization to compress model weights, reducing memory usage from 4.4GB to 1.1GB. This allowed reasonable batch sizes without running out of GPU memory. My base training configuration used mixed-precision training (fp16), the paged AdamW optimizer for memory efficiency, and a cosine learning rate schedule.

### 2.3 Experimental Design

Rather than fine-tuning once with standard settings, I systematically tested five configurations:

**Experiment 1 (Baseline)**: LR=2e-4, batch_size=4, rank=16, epochs=1 → Loss: 1.2463

**Experiment 2 (Lower LR)**: LR=1e-4, batch_size=4, rank=16, epochs=1 → Loss: 1.2347  
*Observation*: Slightly better (1% improvement), smoother loss curve but minimal difference.

**Experiment 3 (Higher Rank)**: LR=2e-4, batch_size=4, rank=32, epochs=1 → Loss: 1.1856  
*Observation*: Best single-epoch result (5% improvement), but doubled training time.

**Experiment 4 (Smaller Batch)**: LR=2e-4, batch_size=2, rank=16, epochs=1 → Loss: 1.2186  
*Observation*: Marginal 2% improvement, noticeably slower training.

**Experiment 5 (More Epochs)**: LR=1.5e-4, batch_size=4, rank=16, epochs=2 → Loss: 1.0823  
*Observation*: **Best performing** with 13.1% improvement over baseline. The model continued learning in the second epoch without overfitting.

---

## 3. Results

### 3.1 Quantitative Metrics

I evaluated all models using BLEU and ROUGE metrics to measure how well generated text matched reference answers:

| Metric | Base Model | Best Fine-tuned | Improvement |
|--------|------------|----------------|-------------|
| BLEU Score | 12.34 | 28.67 | +132% |
| ROUGE-1 | 0.21 | 0.48 | +129% |
| ROUGE-2 | 0.09 | 0.22 | +144% |

The results surprised me—I expected the higher-rank model (Experiment 3) with double the parameters to perform best. Instead, simply training longer with standard settings yielded better results. This suggests that for this dataset size, the constraint wasn't model capacity but getting enough exposure to the training data.

### 3.2 Qualitative Analysis

To understand real-world performance, I tested various medical questions:

**Example 1 (Strong Performance)**: "What is hypertension and how is it treated?"

*Base Model*: "Hypertension is high blood pressure. It can be treated with medication and lifestyle changes."

*Fine-tuned Model*: "Hypertension, or high blood pressure, is a condition where blood pressure in the arteries is consistently elevated (≥140/90 mmHg). Treatment includes: 1) Lifestyle modifications: reduced sodium intake, regular exercise, weight management, and stress reduction. 2) Medications: ACE inhibitors, ARBs, beta-blockers, calcium channel blockers, or diuretics, depending on severity and patient factors. Regular monitoring is essential."

The fine-tuned model provides specific clinical criteria, categorizes treatments clearly, names medication classes, and emphasizes monitoring—exactly the educational information I hoped to provide.

**Example 2 (Weak Performance)**: "What is the typical dosage of metformin?"

*Fine-tuned Model*: "Metformin is commonly prescribed for Type 2 diabetes, usually starting at a low dose and gradually increasing. The dosage depends on individual factors. Always follow your doctor's prescription."

While the model correctly avoids giving specific numbers (good from a safety perspective), it doesn't provide educational information like typical ranges. This reflects limited dosage details in the training data.

**Limitations Observed**: The model struggles with rare diseases (shallow coverage from limited training samples), drug dosages, procedural details, and numerical calculations. It also lacks knowledge of medical advances after 2023.

---

## 4. Deployment

### 4.1 Interface Design

I built the interface using Gradio, which provides a clean chat interface. Key features include chat history, adjustable parameters (temperature, max length), prominent medical disclaimers, and example questions. The disclaimer was critical—I made it highly visible because I never want someone to use this tool instead of seeking proper medical care.

### 4.2 Deployment Challenges

Deploying to Hugging Face Spaces proved more challenging than training. I encountered several issues:

**Python 3.13 Incompatibility**: The app crashed with `ModuleNotFoundError: No module named 'audioop'` because Spaces used Python 3.13 by default, and audioop was removed in that version. I solved this by creating a Dockerfile specifying Python 3.10 and changing from `sdk: gradio` to `sdk: docker`.

**Package Conflicts**: Got `ImportError: cannot import name 'HfFolder'` because newer huggingface-hub versions removed classes that Gradio expected. Through trial and error, I found compatible versions: Gradio 4.29.0 and huggingface-hub 0.20.3.

**CPU Deployment**: Spaces' free tier runs on CPU, but my code assumed GPU with 4-bit quantization. I added device detection to load with or without quantization based on availability. CPU inference is slow (2-3 minutes for first response, 20-30 seconds after), but acceptable for a free demo.

These challenges taught me that deployment involves solving practical problems—version compatibility, resource constraints, library bugs—that aren't covered in ML courses but are crucial for real applications.

---

## 5. Discussion

### 5.1 Key Findings

The most important finding was that **training time mattered more than model capacity** for my setup. Two epochs outperformed double the LoRA rank despite having the same trainable parameters. With 5,000 samples, the model needed more data exposure rather than more parameters. This has practical implications: with limited GPU time, longer training with standard sizes may beat architectural optimization.

I was also pleasantly surprised by 4-bit quantization's effectiveness. Despite 8x compression, the quantized model fine-tuned as effectively as full precision would have, making modern fine-tuning accessible on consumer hardware.

### 5.2 Limitations

I must be honest about limitations:

**Data**: Only 5,000 samples means limited coverage of rare diseases and specialized topics. English-only limits accessibility. No information post-2023.

**Model**: Can't perform calculations, interpret medical images, or cite sources. Tends toward generic answers rather than admitting uncertainty. May generate plausible but incorrect information.

**Evaluation**: BLEU/ROUGE don't capture medical accuracy. No expert review of clinical correctness. No testing for harmful outputs or demographic biases.

**Deployment**: Slow CPU inference. Basic UI without accessibility features. No conversation persistence.

### 5.3 Ethical Considerations

Deploying medical AI raises serious ethical questions. Could someone rely on this instead of seeing a doctor? I mitigated this with clear disclaimers, but can't prevent all misuse. Without clinical expert review, I can't guarantee medical accuracy—the model might generate plausible but wrong information. By training only on English data, I exclude non-English speakers who might benefit most from accessible medical information.

In production, I would want extensive red-teaming, expert medical review, and possibly human oversight for sensitive queries.

### 5.4 Future Work

With more resources, I would: scale to the full 33,000-sample dataset; add multilingual medical data; test larger models (Llama-2-7B, Mistral-7B); implement retrieval-augmented generation to ground responses in verified sources; add source attribution; conduct clinical expert review and user studies; test for biases; and deploy on GPU for faster inference.

---

## 6. Conclusion

This project demonstrated that effective domain-specific language models can be built with limited resources through systematic experimentation and efficient fine-tuning. Starting with TinyLlama and 5,000 medical Q&A samples, I achieved 132% BLEU score improvement through LoRA, establishing that training duration matters more than model capacity for this dataset size.

The deployment phase taught me that production ML involves solving practical challenges—compatibility issues, resource constraints—that aren't covered in courses but are essential for real applications. Most importantly, building medical AI tools requires careful ethical consideration. My chatbot is a step toward accessible health education, but much work remains to ensure such tools are safe, accurate, and equitable.

The complete pipeline—from dataset selection through systematic experimentation to public deployment—represents my growth as an ML practitioner. I'm proud of the final product while remaining realistic about its limitations and the work needed to make medical AI both effective and trustworthy.

---

## References

1. Zhang, P., et al. (2023). "TinyLlama: An Open-Source Small Language Model."
2. Hu, E.J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models."
3. Han, T., et al. (2023). "MedAlpaca: Medical Conversational AI Models."
4. Jin, D., et al. (2021). "What Disease does this Patient Have? Large-scale Medical QA Dataset."
5. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs."

**Project Links**:
- GitHub: https://github.com/karizacharlotte/Domain-Specific-Assistant-via-LLMs-Fine-Tuning
- Colab: https://colab.research.google.com/drive/19vCx0wqRTY5a2hM1hOIEsDlBnCzH6z2f

---

**Word Count**: ~2,000 words
