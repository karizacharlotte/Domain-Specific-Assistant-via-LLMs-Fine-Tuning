# WRITING GUIDE - How to Write in Your Own Voice

## RULE #1: Use "I" statements
❌ BAD: "The model was fine-tuned using LoRA"
✅ GOOD: "I fine-tuned the model using LoRA because it required 99% fewer training parameters"

## RULE #2: Explain your reasoning
❌ BAD: "The learning rate was set to 2e-4"
✅ GOOD: "I started with a learning rate of 2e-4 based on the TinyLlama paper's recommendations, then experimented with lower values to avoid overshooting optimal weights"

## RULE #3: Be specific about YOUR experience
❌ BAD: "Training can be resource-intensive"
✅ GOOD: "My baseline experiment took 23 minutes on Colab's T4 GPU and used 14GB of RAM"

## RULE #4: Admit what you didn't know
❌ BAD: "The approach was carefully designed"
✅ GOOD: "Initially, I tried batch size 8, but this caused out-of-memory errors. After researching gradient accumulation, I reduced to batch size 4"

---

## SENTENCE STARTERS YOU CAN USE (then complete in your voice):

### For Introduction:
- "I became interested in this project when..."
- "After researching medical AI applications, I noticed..."
- "The main challenge I wanted to address was..."
- "My goal was to demonstrate that..."

### For Methodology:
- "I chose TinyLlama because..."
- "To preprocess the data, I..."
- "After experimenting with different configurations, I found..."
- "The reason I used LoRA instead of full fine-tuning was..."

### For Results:
- "The results surprised me because..."
- "I noticed that when I increased X, Y happened..."
- "Comparing experiments 2 and 3, I observed..."
- "The model performed well on X but struggled with Y..."

### For Challenges:
- "The biggest challenge I faced was..."
- "When X happened, I tried..."
- "I spent several hours debugging..."
- "Initially, I misunderstood..."

### For Conclusions:
- "Looking back, I would have..."
- "If I had more time, I would..."
- "The most important lesson I learned was..."
- "This project taught me..."

---

## EXAMPLE PARAGRAPH (good style):

"I chose to fine-tune TinyLlama-1.1B rather than a larger model like Llama-7B for practical reasons. Google Colab's free T4 GPU has 16GB of memory, and when I calculated the memory requirements for full fine-tuning of Llama-7B (approximately 28GB), I realized it wouldn't fit. TinyLlama, at 1.1 billion parameters, could fit comfortably even with 4-bit quantization enabled. Additionally, I found that the smaller model allowed me to run multiple experiments in a single day, which was crucial for comparing different hyperparameter configurations."

**Why this is good:**
- Uses "I" throughout
- Gives specific numbers (16GB, 28GB, 1.1B)
- Explains reasoning (practical constraints)
- Shows the thought process (calculated requirements)
- Connects to the larger goal (multiple experiments)

---

## EXAMPLE PARAGRAPH (bad style - too AI-like):

"The selection of an appropriate base model is crucial for successful fine-tuning. TinyLlama-1.1B-Chat offers an excellent balance of performance and efficiency. With its compact architecture and strong pre-training, it serves as an ideal foundation for domain-specific applications. The model's size enables rapid experimentation while maintaining competitive performance metrics."

**Why this is bad:**
- Generic language ("crucial," "excellent balance")
- No personal experience
- No specific details
- Sounds like marketing copy
- No evidence of actually doing the work

---

## HOW TO DESCRIBE YOUR EXPERIMENTS:

### Template:
"For Experiment [#], I [what you changed] because I wanted to test [hypothesis]. I [kept constant] to isolate the effect of [variable]. The training took [time] and achieved a loss of [number]. I noticed that [observation]. This [confirmed/contradicted] my expectation that [expectation]."

### Example:
"For Experiment 2, I reduced the learning rate from 2e-4 to 1e-4 because I wanted to test whether the model could find better loss minima with smaller weight updates. I kept the batch size and LoRA rank constant to isolate the effect of learning rate. The training took 24 minutes and achieved a loss of 1.2347, slightly better than the baseline's 1.2463. I noticed that the loss curve was smoother with fewer spikes, but the final improvement was minimal. This contradicted my expectation that a lower learning rate would significantly improve performance."

---

## HOW TO DESCRIBE FAILURES (very important!):

### Template:
"When I tried [X], I encountered [problem]. Initially, I thought [wrong assumption]. After [what you did to debug], I realized [actual cause]. To fix this, I [solution]. In hindsight, I should have [lesson learned]."

### Example:
"When I first deployed to Hugging Face Spaces, the app crashed with a 'Module not found: audioop' error. Initially, I thought it was a missing package in requirements.txt. After reading the error carefully, I realized that audioop was removed in Python 3.13, and Spaces was using the latest Python version by default. To fix this, I created a Dockerfile specifying Python 3.10 and changed the Space configuration from 'sdk: gradio' to 'sdk: docker'. In hindsight, I should have checked the Python version compatibility before deploying."

---

## PHRASES TO AVOID (too AI-generated):

❌ "It is noteworthy that..."
❌ "This paper presents..."
❌ "In order to..."
❌ "The results demonstrate that..."
❌ "It can be seen that..."
❌ "Furthermore,..."
❌ "Moreover,..."
❌ "The aforementioned..."
❌ "Cutting-edge..."
❌ "State-of-the-art..."
❌ "Robust performance..."
❌ "Comprehensive analysis..."

## PHRASES TO USE INSTEAD (more natural):

✅ "I found that..."
✅ "My project shows..."
✅ "To..."
✅ "My results show..."
✅ "Looking at the data..."
✅ "Also,..."
✅ "Additionally,..."
✅ "This [specific thing] I mentioned..."
✅ "Recent work..."
✅ "High-performing..."
✅ "Strong results..."
✅ "Detailed comparison..."

---

## DESCRIBING NUMBERS:

### Good:
- "BLEU score improved from 12.34 to 28.67 (132% increase)"
- "Training took 23 minutes per epoch on T4 GPU"
- "I used 5,000 samples (4,500 train, 500 validation)"
- "Final loss decreased from 1.2463 to 1.0823"

### Bad:
- "Significant improvements were observed"
- "Training was efficient"
- "A substantial dataset was utilized"
- "The loss converged well"

---

## DESCRIBING QUALITATIVE RESULTS:

### Good:
"When I tested the question 'What is hypertension?', the base model gave a vague response mentioning 'high blood pressure' but no specifics. My fine-tuned model provided the clinical definition (≥140/90 mmHg) and listed specific treatments like ACE inhibitors. However, when I asked about dosages, both models failed to give accurate information, showing the limitation of fine-tuning without medical dosage data."

### Bad:
"The fine-tuned model demonstrated superior performance on medical queries, providing more detailed and accurate responses compared to the base model across various topics."

---

## FINAL CHECKLIST:

Before submitting each section, ask yourself:
1. Did I use "I" or "my" at least once per paragraph?
2. Did I include specific numbers from MY experiments?
3. Did I explain WHY I made each choice?
4. Did I describe what I actually observed, not what "should" happen?
5. Did I admit when something didn't work?
6. Could someone replicate my work from this description?
7. Does this sound like ME talking, not a textbook?
8. Did I avoid generic phrases like "significant" or "comprehensive"?

---

## TIME-SAVING TIP:

1. Fill in ALL the [FILL IN] sections first (even if rough)
2. Print or have template open side-by-side
3. Go through each section and:
   - Add "I" and "my"
   - Add specific numbers
   - Add "because" or "in order to"
   - Add what you observed
   - Remove generic words
4. Read aloud - does it sound like you explaining to a friend?

---

Remember: A report about YOUR work should sound like YOU. If asked "did you write this?", you should be able to explain every detail because you actually did the work!
