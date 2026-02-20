# Demo Video Script: Medical Assistant Fine-Tuned LLM
**Duration: 7-10 minutes**

---

## **[0:00-0:45] Introduction (45 seconds)**

*[Show title slide with project name]*

"Hi, I'm Charlotte Kariza, and today I'm presenting my Machine Learning project: Fine-Tuning TinyLlama for Medical Information Access.

Access to reliable medical information is a challenge in many communities, especially in rural and underserved areas. While large language models show promise, most require massive computational resources. 

My goal was to create an accessible medical AI assistant using only free GPU resources, systematic experimentation, and efficient fine-tuning techniques. Let me show you what I built."

---

## **[0:45-2:30] Problem & Approach (1 min 45 sec)**

*[Show slides with key points]*

"I started with three main objectives:

**First**, fine-tune a language model on medical Q&A data using parameter-efficient methods that work on Google Colab's free T4 GPU.

**Second**, systematically experiment with different hyperparameters rather than just using default settings.

**Third**, deploy a working chatbot that people can actually use.

For this project, I chose **TinyLlama-1.1B** as my base model—small enough to train on free resources but large enough to learn meaningful patterns. I used **LoRA**, or Low-Rank Adaptation, which lets me train only 8 million parameters instead of the full 1.1 billion.

My training data consisted of **5,000 medical question-answer pairs** from the MedQA dataset, split into 4,500 for training and 500 for validation."

---

## **[2:30-4:30] Experiments & Results (2 minutes)**

*[Show results table/graphs]*

"Rather than fine-tuning once, I ran **five systematic experiments** to understand what actually matters:

**Experiment 1** was my baseline—standard LoRA settings with 1 epoch.

**Experiment 2** tested a lower learning rate—only 1% improvement, not significant.

**Experiment 3** doubled the LoRA rank to increase model capacity—5% improvement, but doubled training time.

**Experiment 4** used smaller batch sizes—marginal 2% improvement.

**Experiment 5** was the winner: I trained for 2 epochs instead of 1, using a learning rate of 1.5e-4. This achieved a **13% improvement over baseline** and was my best-performing configuration.

The key finding? **Training time mattered more than model capacity** for this dataset size.

Compared to the base model, my fine-tuned version achieved:
- **BLEU score: 28.67** versus 12.34—a **132% improvement**
- **ROUGE-1: 0.48** versus 0.21—**129% improvement**

These metrics show the model learned to generate much more relevant medical information."

---

## **[4:30-7:00] Live Demo (2 min 30 sec)**

*[Screen recording of the chatbot interface]*

"Now let me show you the deployed chatbot in action. I built this interface using Gradio and deployed it on Hugging Face Spaces.

**[Type first question]** Let's ask: 'What is hypertension and how is it treated?'

*[Wait for response]*

Notice how the model provides:
- A clear definition with clinical criteria
- Structured treatment approaches
- Specific medication classes
- Practical advice about monitoring

**[Type second question]** Let's try something different: 'Explain the difference between bacteria and viruses.'

*[Wait for response]*

The model gives a comprehensive explanation covering structure, reproduction, treatment implications, and examples—exactly the educational information we want.

**[Type third question]** One more: 'What are the symptoms of diabetes?'

*[Wait for response]*

The model lists symptoms with medical terminology, distinguishes between Type 1 and Type 2, and importantly, advises consulting a healthcare provider.

You'll notice the prominent **medical disclaimer**—this is crucial because this tool is for education only, not medical advice."

---

## **[7:00-8:30] Deployment Challenges (1 min 30 sec)**

*[Show code snippets or deployment logs]*

"Deployment was actually harder than training the model. I encountered several challenges:

**Python 3.13 compatibility issues**—I had to create a custom Dockerfile to use Python 3.10.

**Package version conflicts**—finding compatible versions of Gradio and huggingface-hub required extensive testing.

**CPU versus GPU**—Spaces' free tier runs on CPU, so I had to add logic to detect the device and disable quantization for CPU inference.

The result: responses take about 20-30 seconds on CPU hosting, which isn't ideal but acceptable for an educational demo.

These challenges taught me that deployment is a critical part of ML engineering that often gets overlooked in coursework."

---

## **[8:30-9:30] Limitations & Ethics (1 minute)**

*[Show limitation points]*

"I need to be transparent about limitations:

**Data limitations**: Only 5,000 samples means limited coverage of rare diseases and specialized topics.

**Model limitations**: The model can't perform calculations, interpret medical images, or cite sources. It may occasionally generate plausible but incorrect information.

**Ethical considerations**: There's risk of misuse—someone might rely on this instead of seeing a doctor. That's why the medical disclaimer is so prominent.

In production, this would need:
- Expert medical review
- Extensive testing for harmful outputs
- Bias testing across demographics
- Possibly human oversight for sensitive queries"

---

## **[9:30-10:00] Conclusion (30 seconds)**

*[Show final slide with links]*

"This project demonstrates that effective domain-specific language models can be built with limited resources through systematic experimentation and efficient techniques.

I achieved a **132% improvement in BLEU scores** using only free GPU access, and deployed a working chatbot that people can use for medical education.

More importantly, this experience taught me about the complete ML pipeline—from data preprocessing through deployment—and the ethical responsibilities that come with building AI tools for healthcare.

Thank you for watching. The code, model, and live demo are all publicly available at the links shown here."

*[Show GitHub, Colab, and Hugging Face Space links]*

---

## **Presentation Tips:**

1. **Pacing**: Speak clearly and pause between sections. Don't rush.

2. **Screen Recording**: Pre-record the chatbot demo to avoid waiting for slow CPU inference during the live presentation.

3. **Visuals**: Use simple, clear slides with key metrics and bullet points.

4. **Confidence**: You've done excellent work—let your enthusiasm show!

5. **Timing Check**: Practice to ensure you stay within 7-10 minutes. You can adjust the demo section based on response times.

6. **Backup Plan**: If live demo fails, have screenshots of successful responses ready.

7. **Energy**: Maintain good energy throughout—smile and make eye contact with the camera.

8. **Transitions**: Use smooth transitions between sections: "Now that we've seen the results, let me show you..."

---

## **Visual Aids Checklist:**

- [ ] Title slide with project name and your name
- [ ] Problem statement slide (healthcare access challenges)
- [ ] Approach overview (TinyLlama, LoRA, dataset size)
- [ ] Experiments comparison table
- [ ] Results metrics (BLEU, ROUGE improvements)
- [ ] Live demo screen recording
- [ ] Deployment challenges bullet points
- [ ] Limitations overview
- [ ] Final slide with links (GitHub, Colab, HuggingFace)

---

## **Alternative 5-Minute Version** (If Time Constrained):

For a shorter 5-minute version, compress as follows:
- **Introduction**: 30 seconds
- **Problem & Approach**: 1 minute
- **Experiments & Results**: 1 minute (focus on best result only)
- **Live Demo**: 1.5 minutes (show 1-2 examples only)
- **Conclusion**: 1 minute (combine deployment challenges, limitations, and closing)

---

**Good luck with your demo! You've built something impressive! 🎓**
