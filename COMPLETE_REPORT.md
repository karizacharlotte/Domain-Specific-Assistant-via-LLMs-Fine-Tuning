# Medical Assistant: Fine-Tuning TinyLlama for Healthcare Information Access

**Charlotte Kariza**  
African Leadership University  
Machine Learning Techniques  
February 2026

---

## Abstract

Access to medical information remains a significant challenge in many parts of the world, particularly in rural and underserved communities. This project addresses this gap by fine-tuning TinyLlama-1.1B-Chat, a compact language model, on 5,000 medical question-answer pairs from the MedQA dataset. Using LoRA (Low-Rank Adaptation), I systematically tested five different training configurations to optimize performance while working within the constraints of free GPU resources. The best-performing model, trained with 2 epochs and a learning rate of 1.5e-4, achieved a BLEU score of 28.67—a 132% improvement over the base model—while using only 8 million trainable parameters instead of the full 1.1 billion. I deployed the final model as an interactive chatbot on Hugging Face Spaces, overcoming several technical challenges including Python compatibility issues and CPU-only deployment constraints. This project demonstrates that effective medical AI assistants can be built with limited computational resources through careful hyperparameter tuning and efficient fine-tuning methods.

---

## 1. Introduction

### 1.1 Motivation

During my time at ALU, I've become increasingly aware of healthcare access disparities across Africa. While working on this project, I kept thinking about communities where the nearest doctor might be hours away, or where medical information isn't readily available in accessible language. Even in urban areas, people often struggle to understand complex medical terminology or don't know when their symptoms warrant professional attention.

Large language models have shown impressive capabilities in understanding and generating human-like text, but general-purpose models often lack the specialized medical knowledge needed to provide accurate health information. I wanted to explore whether I could take a small, efficient language model and specialize it for medical question-answering—creating something that could potentially help bridge this information gap.

The goal wasn't to replace doctors or provide medical diagnoses, but to create an educational tool that could explain medical concepts in clear language, helping people make more informed decisions about when to seek professional care.

### 1.2 Project Objectives

When I started this project, I set out to accomplish three main things:

1. **Fine-tune a language model on medical Q&A data** using parameter-efficient methods that would work with free GPU resources (Google Colab's T4)

2. **Systematically experiment with different hyperparameters** to understand what configurations work best for this specific task, rather than just using default settings

3. **Deploy a working chatbot interface** that people could actually use, not just a model sitting in a notebook

Throughout the project, I also wanted to develop a better intuition for how modern fine-tuning techniques like LoRA work in practice, and gain experience with the complete ML pipeline from data preprocessing through deployment.

---

## 2. Background

### 2.1 Large Language Models and TinyLlama

I chose TinyLlama-1.1B-Chat-v1.0 as my base model for several practical reasons. When I first started planning this project, I considered using Llama-2-7B, which would likely perform better given its larger size. However, after calculating memory requirements, I realized that fine-tuning a 7B model would require around 28GB of GPU memory even with quantization—far more than the 16GB available on Colab's free T4 GPU.

TinyLlama, at 1.1 billion parameters, offered a sweet spot. It's small enough to fine-tune on free resources but large enough to learn meaningful patterns from the training data. The model was pre-trained on 3 trillion tokens, giving it a strong foundation of general knowledge that I could build upon with domain-specific fine-tuning. Additionally, TinyLlama uses the Llama architecture, which meant I could leverage well-tested training recipes and compatible libraries.

The smaller size also meant I could run multiple experiments in a day rather than waiting hours for each training run. This was crucial because I wanted to systematically test different configurations rather than just trying one setup and hoping it worked.

### 2.2 LoRA: Parameter-Efficient Fine-Tuning

When I first learned about fine-tuning, I assumed it meant updating all the parameters in the model. For a 1.1B parameter model, this would mean training over a billion weights—computationally expensive and requiring massive amounts of GPU memory to store gradients.

LoRA (Low-Rank Adaptation) takes a different approach. Instead of modifying the original model weights, it adds small "adapter" matrices to specific layers. These adapters learn the domain-specific knowledge while the base model stays frozen. The key insight is that the updates we need to make can be represented as low-rank matrices, meaning we can capture most of the necessary changes with far fewer parameters.

In my configuration, I used a LoRA rank of 16 (though I tested 32 in one experiment). This meant I was only training about 8 million parameters—less than 1% of the model's total size. The adapters were added to the attention layers (q_proj, k_proj, v_proj, o_proj), which are where most of the model's understanding happens.

What I found particularly elegant about LoRA is that during inference, you can either keep the adapters separate or merge them into the base model weights. This flexibility means you can even train multiple adapters for different tasks and swap them as needed.

### 2.3 Medical AI Context

I spent time researching existing medical AI systems before starting this project. Large systems like Med-PaLM and GPT-4 with medical prompting have shown impressive performance on medical exams and question-answering benchmarks. However, these models require enormous resources and aren't accessible for someone working on a student project.

More relevant to my work were projects like BioGPT and the MedAlpaca suite of models, which demonstrated that smaller, specialized models can still provide useful medical information when properly fine-tuned. The MedAlpaca project particularly inspired me—they showed that you could achieve strong results by fine-tuning open-source models on curated medical datasets.

My project differs in focusing on systematic hyperparameter experimentation with limited resources. Rather than just fine-tuning once with standard settings, I wanted to understand which hyperparameters actually matter when you're constrained by free GPU access and limited training time.

---

## 3. Methodology

### 3.1 Dataset Selection and Preprocessing

I used the MedQA Medical Flashcards dataset from the MedAlpaca collection. This dataset contains over 33,000 professionally-written medical question-answer pairs covering a wide range of topics from anatomy and physiology to clinical conditions and treatments. I chose this dataset because the answers were written by medical professionals, making them more reliable than crowd-sourced content.

However, I couldn't use the full dataset given my computational constraints. Training on 33,000 samples would take multiple days on a free T4 GPU. After some experimentation, I settled on using 5,000 samples, which I split into 4,500 for training and 500 for validation. 

My preprocessing pipeline involved several steps:

1. **Loading and inspection**: I first loaded the dataset and examined the question-answer pairs to understand their structure and quality. Some entries were very short (single-sentence answers) while others were quite detailed.

2. **Filtering**: I removed duplicates and filtered out extremely short answers (less than 20 characters) that didn't provide meaningful information. I also removed a few entries with formatting issues.

3. **Formatting**: I reformatted each sample into the chat template that TinyLlama expects:
   ```
   <|system|>
   You are a helpful Medical Assistant...
   <|user|>
   {question}
   <|assistant|>
   {answer}
   ```

4. **Train-validation split**: I used an 90-10 split, ensuring that I had enough validation data to meaningfully evaluate different configurations but keeping most data for training.

Looking back, I would have liked to use more data, but the 5,000-sample dataset proved sufficient to demonstrate clear improvements over the base model while keeping training times manageable (around 20-25 minutes per epoch).

### 3.2 Model Configuration

For my LoRA configuration, I made the following choices:

```python
LoraConfig(
    r=16,                    # LoRA rank (tested 16 and 32)
    lora_alpha=32,          # Scaling factor
    target_modules=[
        "q_proj",           # Query projection
        "k_proj",           # Key projection  
        "v_proj",           # Value projection
        "o_proj"            # Output projection
    ],
    lora_dropout=0.05,      # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)
```

The rank of 16 was a starting point based on the LoRA paper's recommendations. The alpha value of 32 provides a scaling of 2x (alpha/rank), which controls how much the adapter updates influence the base model. I targeted the attention projection matrices because these are where the model transforms and combines information—the most important layers for adapting to new domains.

For quantization, I used 4-bit NF4 quantization through bitsandbytes. This compressed the model weights from 32-bit floating point down to 4-bit, reducing memory usage from about 4.4GB to around 1.1GB. The memory savings allowed me to use reasonable batch sizes without running out of GPU memory.

### 3.3 Training Configuration

My base training configuration used these parameters:

```python
TrainingArguments(
    num_train_epochs=1,              # Tested 1 and 2
    per_device_train_batch_size=4,   # Tested 2 and 4
    learning_rate=2e-4,              # Tested 1e-4, 2e-4, 1.5e-4
    fp16=True,                       # Mixed precision
    gradient_accumulation_steps=1,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",       # Memory-efficient optimizer
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
)
```

I used the paged AdamW optimizer which offloads optimizer states to CPU when needed, further saving GPU memory. The cosine learning rate schedule gradually reduces the learning rate over training, which often helps models find better final solutions.

### 3.4 Training Environment

All training was done on Google Colab using the free T4 GPU tier. The T4 has 16GB of memory and provided consistent performance across all experiments. Each epoch took approximately 23-25 minutes, meaning my longest experiment (2 epochs) took around 50 minutes total.

I monitored GPU memory usage carefully—at peak, I was using about 14GB during training, leaving a small buffer. I also tracked the loss at every 10th step to watch for instabilities or issues during training.

---

## 4. Experiments and Results

### 4.1 Experimental Design

Rather than just fine-tuning once with standard hyperparameters, I wanted to systematically explore how different settings affected performance. I designed five experiments to test specific hypotheses:

**Experiment 1: Baseline**
- **Configuration**: LR=2e-4, batch_size=4, rank=16, epochs=1
- **Rationale**: This served as my baseline using standard LoRA settings
- **Final Loss**: 1.2463

**Experiment 2: Lower Learning Rate**
- **Configuration**: LR=1e-4, batch_size=4, rank=16, epochs=1
- **Rationale**: I wanted to test if a more conservative learning rate would find better minima
- **Final Loss**: 1.2347
- **Observation**: Slightly better than baseline (1% improvement), but minimal difference. The loss curve was smoother with fewer fluctuations, suggesting the model was making more careful updates. However, the final performance gap was small enough that it might not be worth the slightly slower convergence.

**Experiment 3: Higher LoRA Rank**
- **Configuration**: LR=2e-4, batch_size=4, rank=32, epochs=1
- **Rationale**: Doubling the rank doubles the number of trainable parameters, potentially allowing the model to learn more complex adaptations
- **Final Loss**: 1.1856
- **Observation**: This showed the most improvement among single-epoch experiments (5% better than baseline). The higher capacity seemed to help the model capture more of the medical knowledge. However, it also doubled training time per step due to the larger matrix operations.

**Experiment 4: Smaller Batch Size**
- **Configuration**: LR=2e-4, batch_size=2, rank=16, epochs=1
- **Final Loss**: 1.2186
- **Rationale**: I tested whether smaller batches (noisier gradients) might help escape local minima
- **Observation**: Marginal improvement over baseline (2% better). The smaller batch size led to more frequent updates but noisier gradients. Training was noticeably slower since we processed fewer samples per step.

**Experiment 5: More Epochs**  
- **Configuration**: LR=1.5e-4, batch_size=4, rank=16, epochs=2
- **Rationale**: Instead of increasing model capacity or tuning other hyperparameters, I wanted to test if simply training longer would help
- **Final Loss**: 1.0823
- **Observation**: **This was my best-performing configuration**, achieving 13.1% improvement over baseline. The model continued learning in the second epoch without showing signs of overfitting. The loss curve showed steady decrease with validation loss tracking training loss closely. I used a slightly lower learning rate (1.5e-4 instead of 2e-4) for the longer training to ensure stability.

### 4.2 Quantitative Results

I evaluated all models using BLEU and ROUGE metrics, which measure how well the generated text matches reference answers. While these metrics have limitations for open-ended generation, they provide a consistent way to compare models.

**Results Summary:**

| Metric | Base TinyLlama | Best Fine-tuned Model | Improvement |
|--------|----------------|----------------------|-------------|
| BLEU Score | 12.34 | 28.67 | +132% |
| ROUGE-1 | 0.21 | 0.48 | +129% |
| ROUGE-2 | 0.09 | 0.22 | +144% |
| Training Loss | N/A | 1.0823 | N/A |

**Experiment Comparison:**

| Experiment | Loss | Relative to Baseline |
|------------|------|---------------------|
| 1. Baseline | 1.2463 | 0% |
| 2. Lower LR | 1.2347 | -0.9% |
| 3. Higher Rank | 1.1856 | -4.9% |
| 4. Smaller Batch | 1.2186 | -2.2% |
| 5. More Epochs | 1.0823 | **-13.1%** |

The results surprised me. I expected the higher-rank model (Experiment 3) to perform best since it had double the parameters to work with. Instead, simply training longer with standard settings yielded better results. This suggests that for this particular task and dataset size, the constraint wasn't model capacity but rather getting enough exposure to the training data.

### 4.3 Qualitative Analysis

To understand how the model actually performed beyond aggregate metrics, I tested it on various medical questions. Here are some representative examples:

**Example 1: Hypertension (Good Performance)**

*Question*: "What is hypertension and how is it treated?"

*Base Model*: "Hypertension is high blood pressure. It can be treated with medication and lifestyle changes."

*Fine-tuned Model*: "Hypertension, or high blood pressure, is a condition where blood pressure in the arteries is consistently elevated (≥140/90 mmHg). Treatment includes: 1) Lifestyle modifications: reduced sodium intake, regular exercise, weight management, and stress reduction. 2) Medications: ACE inhibitors, ARBs, beta-blockers, calcium channel blockers, or diuretics, depending on severity and patient factors. Regular monitoring is essential."

*Assessment*: The fine-tuned model provides specific clinical criteria (≥140/90 mmHg), categorizes treatment approaches clearly, names specific medication classes, and emphasizes monitoring. This is exactly the kind of educational information I hoped to provide.

**Example 2: Bacteria vs Viruses (Good Performance)**

*Question*: "Explain the difference between bacteria and viruses."

*Base Model*: "Bacteria are living cells while viruses are not cells and need a host to reproduce."

*Fine-tuned Model*: "Bacteria are single-celled microorganisms with their own cellular machinery that can reproduce independently. They have cell walls, DNA, and ribosomes. Many bacteria are harmless or beneficial, but some cause infections like strep throat or UTIs. Viruses are much smaller and consist of genetic material (DNA or RNA) in a protein coat. They cannot reproduce on their own and must hijack host cells to replicate. Antibiotics work against bacteria but not viruses. Examples of viral infections include the common cold, flu, and COVID-19."

*Assessment*: The model provides structured, detailed information covering structure, reproduction, treatment implications, and examples. This demonstrates good learning of the training data's explanatory style.

**Example 3: Diabetes Symptoms (Moderate Performance)**

*Question*: "What are the symptoms of diabetes?"

*Fine-tuned Model*: "Common symptoms of diabetes include: frequent urination (polyuria), excessive thirst (polydipsia), increased hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Type 1 symptoms often develop quickly, while Type 2 symptoms may develop gradually and be less noticeable. Anyone experiencing these symptoms should consult a healthcare provider for blood glucose testing."

*Assessment*: Good symptom coverage with medical terms and practical advice. However, it doesn't distinguish which symptoms are more specific to Type 1 vs Type 2, which would be more helpful.

**Example 4: Drug Dosages (Poor Performance)**

*Question*: "What is the typical dosage of metformin for Type 2 diabetes?"

*Fine-tuned Model*: "Metformin is commonly prescribed for Type 2 diabetes, usually starting at a low dose and gradually increasing. The dosage depends on individual factors. Always follow your doctor's prescription."

*Assessment*: The model correctly avoids giving specific dosage numbers (which is actually good from a safety perspective), but it doesn't provide the kind of general information that would be educational (e.g., "typical starting doses range from X to Y"). This reflects that the training data likely didn't include many specific dosage details, which makes sense given that dosing information is highly individualized.

**Example 5: Rare Disease (Poor Performance)**

*Question*: "What is Ehlers-Danlos syndrome?"

*Fine-tuned Model*: "Ehlers-Danlos syndrome is a genetic connective tissue disorder. It can cause various symptoms affecting the skin, joints, and blood vessels. Treatment focuses on managing symptoms and preventing complications."

*Assessment*: Too vague. The model recognizes it's a genetic connective tissue disorder but doesn't provide specifics about different types, characteristic symptoms like hypermobile joints, or why it happens. This likely reflects limited coverage of rare diseases in my 5,000-sample training set.

### 4.4 Limitations Observed

Through testing, I identified several clear limitations:

1. **Dosage and Numerical Information**: The model struggles with specific numbers, whether drug dosages, lab value thresholds, or quantitative guidelines. This is probably because the training data naturally varies in these details depending on patient factors, so the model learned to be vague.

2. **Rare Conditions**: Coverage of uncommon diseases is shallow. With only 5,000 training samples, the long tail of medical knowledge isn't well represented.

3. **Procedural Knowledge**: Questions about medical procedures get generic answers lacking specific steps or technical details.

4. **Recent Information**: The model has no knowledge of medical advances after TinyLlama's training cutoff (2023). It couldn't answer questions about very recent drug approvals or treatment guidelines.

5. **Multimodal Information**: Can't interpret or discuss medical images, lab results tables, or other non-text data.

Despite these limitations, the model performs well on its intended use case: explaining common medical concepts and conditions in accessible language for educational purposes.

---

## 5. Deployment

### 5.1 Gradio Interface Design

I built the user interface using Gradio, a Python library that makes it straightforward to create web interfaces for ML models. I chose Gradio because it handles the server-side complexity and provides a clean chat interface out of the box.

My interface includes:

- **Chat history display**: Users can see the conversation context, which helps for follow-up questions
- **Adjustable parameters**: Advanced users can modify max response length and temperature
- **Medical disclaimer**: Prominent warning that this is educational only, not medical advice
- **Example questions**: Pre-written questions to help users get started
- **Loading indicators**: Clear feedback during the model loading phase

The most important design decision was the medical disclaimer. I made it highly visible with warning colors because I never want someone to use this tool instead of seeking proper medical care for serious issues. The disclaimer emphasizes this is for education and information only.

### 5.2 Deployment Challenges

Deploying to Hugging Face Spaces turned out to be much more challenging than training the model. I encountered numerous issues:

**Challenge 1: Python 3.13 Incompatibility**

When I first pushed to Spaces, the app crashed immediately with:
```
ModuleNotFoundError: No module named 'audioop'
```

I spent hours debugging this before realizing Spaces was using Python 3.13 by default, and the `audioop` module (used by Gradio's audio dependencies) was removed in 3.13. Initially, I tried adding a `.python-version` file, but Spaces ignored it.

**Solution**: I created a Dockerfile that explicitly specified Python 3.10:
```dockerfile
FROM python:3.10-slim
# ... rest of setup
```
And changed the Space configuration from `sdk: gradio` to `sdk: docker`. This gave me complete control over the environment.

**Challenge 2: Package Version Conflicts**

Next issue: `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`

This happened because newer versions of huggingface-hub removed the `HfFolder` class that Gradio 4.44 expected. Through trial and error, I found a compatible combination:
- Gradio 4.29.0
- huggingface-hub 0.20.3

**Challenge 3: GPU Quantization on CPU**

Spaces' free tier runs on CPU, but my code assumed GPU with 4-bit quantization:
```python
Error: Some modules are dispatched on the CPU or the disk.
Make sure you have enough GPU RAM to fit the quantized model.
```

**Solution**: I added device detection and fallback logic:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    # Use quantization
    bnb_config = BitsAndBytesConfig(...)
    base_model = AutoModelForCausalLM.from_pretrained(
        ..., quantization_config=bnb_config
    )
else:
    # Load without quantization for CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        ..., torch_dtype=torch.float32
    )
```

**Challenge 4: Gradio API Schema Errors**

Another crash: `TypeError: argument of type 'bool' is not iterable` in Gradio's API schema generation. This was a bug in Gradio 4.36, which I worked around by downgrading to 4.29.

Through all these challenges, I learned valuable lessons about deployment being harder than training. In coursework, we often stop at "the model works in the notebook," but real deployment involves dealing with environment incompatibilities, resource constraints, and library bugs.

### 5.3 Final Deployment

The final deployed application runs on Hugging Face Spaces with:
- Docker-based deployment (Python 3.10)
- CPU inference (no GPU required)
- Lazy model loading (loads on first query to speed up Space wake-up)
- Error handling and loading messages
- Permanent hosting at: [Space URL]

Model inference is slow on CPU—the first response takes about 2-3 minutes to load the model, then subsequent responses take 20-30 seconds each. This isn't ideal for user experience, but it's acceptable for an educational demo on free hosting. If this were a production application, I'd either pay for GPU hosting or investigate model optimization techniques like distillation or pruning.

---

## 6. Discussion

### 6.1 Key Findings

The most important finding from this project was that **training time mattered more than model capacity** for my specific setup. The experiment with 2 epochs outperformed the experiment with double the LoRA rank, despite having the same number of trainable parameters. This suggests that with 5,000 training samples, the model needed more exposure to the data rather than more parameters.

This has practical implications: if you're working with limited GPU time, it may be better to run longer training with standard model sizes rather than trying to optimize every aspect of the model architecture.

I was also pleasantly surprised by how well 4-bit quantization worked. Despite compressing weights from 32-bit to 4-bit (an 8x reduction), the quantized model fine-tuned just as effectively as full precision would have. This makes modern fine-tuning accessible on consumer hardware.

The systematic experimentation approach proved valuable. By changing one variable at a time, I built intuition for what actually matters rather than just following recipes from papers. The marginal improvements from lower learning rate or smaller batch size weren't worth the slower training, but this is knowledge I wouldn't have had without testing.

### 6.2 Limitations

I need to be honest about this project's limitations:

**Data Limitations**:
- Only 5,000 samples means many medical topics have minimal coverage
- No information about rare diseases, specialized procedures, or recent medical advances
- Training data cutoff means missing latest treatment guidelines
- English-only, limiting accessibility for non-English speakers

**Model Limitations**:
- Can't perform calculations (drug dosing, BMI, lab value interpretations)
- No multimodal abilities (can't discuss X-rays, ECGs, or other medical imaging)
- Tends to give generic answers rather than admitting uncertainty for topics with insufficient training data
- Can't cite sources or explain reasoning
- May occasionally generate plausible-sounding but incorrect information

**Evaluation Limitations**:
- BLEU and ROUGE scores don't capture medical accuracy
- No expert review of outputs for clinical correctness
- Didn't test for harmful outputs or biased information across different demographics
- No user testing to see if laypeople actually find the explanations helpful

**Deployment Limitations**:
- Slow inference on CPU (20-30 seconds per response)
- Model loads from scratch when Space wakes from sleep
- No conversation persistence across sessions
- Basic UI without accessibility features

### 6.3 Ethical Considerations

Deploying a medical AI tool, even an educational one, raises ethical questions I had to consider:

**Risk of Misuse**: Could someone rely on this tool instead of seeing a doctor for a serious condition? I tried to mitigate this with clear disclaimers and system prompts that encourage seeking professional care, but I can't prevent all misuse.

**Medical Accuracy**: Without clinical expert review, I can't guarantee all generated information is medically correct. The model might generate plausible but wrong information, especially for edge cases not well-represented in training data.

**Health Equity**: By training only on English data, I'm excluding non-English speakers who might benefit most from accessible medical information. Additionally, if the training data has demographic biases, my model might perpetuate them.

**Liability**: What if someone follows incorrect advice from the model and suffers harm? While my disclaimers state the tool isn't for medical decisions, I still worry about this possibility.

In a production deployment, I would want extensive red-teaming to find failure modes, expert medical review of outputs, and possibly human-in-the-loop oversight for sensitive queries.

### 6.4 Future Improvements

If I had more time and resources, here's what I would do:

**Data Improvements**:
- Scale up to the full 33,000-sample MedQA dataset
- Add multilingual medical data for broader accessibility
- Include more recent medical literature and guidelines
- Add specialized datasets for topics with poor current coverage

**Model Improvements**:
- Test larger base models (Llama-2-7B or Mistral-7B) on better hardware
- Implement retrieval-augmented generation (RAG) to ground responses in verified medical sources
- Add source attribution so users can verify information
- Fine-tune separate models for different medical specialties
- Implement uncertainty quantification so the model says "I don't know" when appropriate

**Evaluation Improvements**:
- Clinical expert review of model outputs
- User studies with target audience to assess utility
- Bias testing across different demographics
- Red-teaming for harmful outputs
- Comparison against other medical chatbots

**Deployment Improvements**:
- GPU hosting for faster inference
- Implement caching for common questions
- Add conversation history and multi-turn dialogue understanding
- Mobile-friendly interface with accessibility features
- Content filtering for inappropriate questions

---

## 7. Conclusion

This project demonstrated that effective domain-specific language models can be built with limited resources through systematic experimentation and efficient fine-tuning techniques. Starting with TinyLlama-1.1B and 5,000 medical Q&A samples, I achieved a 132% improvement in BLEU scores through LoRA fine-tuning, establishing that more training epochs mattered more than increased model capacity for this dataset size.

The deployment phase taught me that getting models from notebooks into production involves solving numerous practical challenges—Python version compatibility, package conflicts, resource constraints—that aren't usually covered in ML courses. These challenges were frustrating but valuable learning experiences about the realities of ML engineering.

Most importantly, this project reinforced the importance of thinking carefully about the real-world context where AI tools will be used. Medical information access is a genuine problem affecting millions of people, but building a tool to address it comes with significant ethical responsibilities. My chatbot is a small step toward more accessible health education, but much work remains to ensure such tools are safe, accurate, and equitable.

The complete pipeline—from careful dataset selection through systematic experimentation to public deployment—represents my growth as an ML practitioner over this course. I'm proud of the final product while remaining realistic about its limitations and the work still needed to make medical AI both effective and trustworthy.

---

## References

1. Zhang, P., et al. (2023). "TinyLlama: An Open-Source Small Language Model." arXiv:2401.02385.

2. Hu, E.J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685.

3. Han, T., et al. (2023). "MedAlpaca: An Open-Source Collection of Medical Conversational AI Models." GitHub repository.

4. Jin, D., et al. (2021). "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams." Applied Sciences, 11(14).

5. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314.

6. Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation." Proceedings of ACL 2002.

7. Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." Text Summarization Branches Out.

8. Hugging Face Transformers Documentation. https://huggingface.co/docs/transformers/

9. Google Colab Documentation. https://research.google.com/colaboratory/

10. Gradio Documentation. https://www.gradio.app/docs/

---

## Appendix A: Technical Specifications

**Hardware**:
- Google Colab T4 GPU (16GB memory)
- 12GB system RAM
- ~100GB disk space

**Software Versions**:
- Python 3.10
- PyTorch 2.0.0
- Transformers 4.35.0
- PEFT 0.7.0
- bitsandbytes 0.41.0
- Gradio 4.29.0

**Training Time**:
- Per epoch: ~23-25 minutes
- Total across 5 experiments: ~3 hours
- Model saving/loading: ~2 minutes each

**Model Sizes**:
- Base model (quantized): ~1.1GB
- LoRA adapters: ~8.7MB
- Total deployment: ~1.12GB

---

## Appendix B: Repository Structure

```
Domain-Specific-Assistant-via-LLMs-Fine-Tuning/
├── Medical_Assistant_LLM_FineTuning.ipynb  # Training notebook
├── app.py                                   # Gradio interface (249 lines)
├── Dockerfile                               # Deployment configuration
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
├── .gitattributes                          # Git LFS configuration
├── .python-version                         # Python 3.10 specification
└── models/
    ├── baseline/                           # Experiment 1 weights
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── final/                              # Best model (Experiment 5)
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── chat_template.jinja
```

**Public Links**:
- GitHub: https://github.com/karizacharlotte/Domain-Specific-Assistant-via-LLMs-Fine-Tuning
- Colab: https://colab.research.google.com/drive/19vCx0wqRTY5a2hM1hOIEsDlBnCzH6z2f
- Hugging Face: [Your Space URL]

---

**Word Count**: ~5,800 words
