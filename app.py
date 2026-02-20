"""
Medical Assistant Chatbot - Gradio Interface
Deployed on Hugging Face Spaces
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Try to import BitsAndBytesConfig for GPU quantization
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("BitsAndBytes not available. Running in CPU mode without quantization.")

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "models/final"
HF_TOKEN = os.getenv("HF_TOKEN", None)  # Hugging Face token from environment variable

# Initialize model and tokenizer
def load_model():
    """Load the base model and apply the fine-tuned adapter"""
    print("Loading model...")
    
    # Check if CUDA is available and quantization is supported
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_quantization = device == "cuda" and QUANTIZATION_AVAILABLE
    print(f"Using device: {device}, Quantization: {use_quantization}")
    
    if use_quantization:
        # Configure 4-bit quantization for efficient inference (GPU only)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
    else:
        # Load model without quantization for CPU
        print("Loading model without quantization (CPU mode)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            token=HF_TOKEN,
            low_cpu_mem_usage=True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, token=HF_TOKEN)
    model.eval()
    
    # Move to device if CPU
    if device == "cpu":
        model = model.to(device)
    
    print("Model loaded successfully!")
    return model, tokenizer

# Global variables (loaded once)
model, tokenizer = None, None

def initialize():
    """Initialize model on first run"""
    global model, tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

def format_prompt(message: str, chat_history: list) -> str:
    """Format the prompt with chat history for the model"""
    prompt = "<|system|>\nYou are a helpful Medical Assistant. Provide accurate, clear medical information. Always remind users to consult healthcare professionals for medical advice.</s>\n"
    
    # Add chat history
    for user_msg, assistant_msg in chat_history:
        prompt += f"<|user|>\n{user_msg}</s>\n"
        prompt += f"<|assistant|>\n{assistant_msg}</s>\n"
    
    # Add current message
    prompt += f"<|user|>\n{message}</s>\n<|assistant|>\n"
    
    return prompt

def generate_response(message: str, chat_history: list, max_length: int = 512, temperature: float = 0.7) -> str:
    """Generate response from the medical assistant model"""
    try:
        initialize()
        
        # Format prompt with history
        prompt = format_prompt(message, chat_history)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and extract only the new response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("<|assistant|>")[-1].strip()
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try again or contact support if the issue persists."

def chat_interface(message: str, history: list, max_length: int, temperature: float):
    """Main chat interface function"""
    if not message.strip():
        return history, history
    
    # Generate response
    response = generate_response(message, history, max_length, temperature)
    
    # Update history
    history.append((message, response))
    
    return history, history

# Custom CSS for better styling
custom_css = """
#warning {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
.disclaimer {
    color: #856404;
    font-weight: bold;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Medical Assistant Chatbot") as demo:
    gr.Markdown("# 🏥 Medical Assistant Chatbot")
    gr.Markdown("Ask medical questions and get informed responses powered by fine-tuned AI.")
    
    gr.Markdown("*⚠️ Note: Running on CPU. Model loads on first message and may take 2-3 minutes for first response.*")
    
    # Disclaimer
    gr.HTML("""
        <div id="warning">
            <p class="disclaimer">⚠️ MEDICAL DISCLAIMER</p>
            <p>This AI assistant is for educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.</p>
        </div>
    """)
    
    # Chat interface
    chatbot = gr.Chatbot(
        label="Chat History",
        height=400,
        show_label=True,
        avatar_images=None,
    )
    
    state = gr.State([])
    
    with gr.Row():
        msg = gr.Textbox(
            label="Your Message",
            placeholder="Ask a medical question... (e.g., 'What is hypertension?')",
            lines=2,
            scale=4,
        )
        
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear Chat", scale=1)
    
    # Advanced settings
    with gr.Accordion("⚙️ Advanced Settings", open=False):
        max_length_slider = gr.Slider(
            minimum=128,
            maximum=1024,
            value=512,
            step=128,
            label="Max Response Length",
            info="Maximum tokens to generate"
        )
        temperature_slider = gr.Slider(
            minimum=0.1,
            maximum=1.5,
            value=0.7,
            step=0.1,
            label="Temperature",
            info="Higher = more creative, Lower = more focused"
        )
    
    # Example questions
    gr.Examples(
        examples=[
            "What is hypertension and how is it treated?",
            "Explain the difference between bacteria and viruses.",
            "What are the symptoms of diabetes?",
            "How does the immune system work?",
            "What is the purpose of vaccination?",
        ],
        inputs=msg,
        label="Example Questions"
    )
    
    # Event handlers
    submit_btn.click(
        fn=chat_interface,
        inputs=[msg, state, max_length_slider, temperature_slider],
        outputs=[chatbot, state],
    ).then(
        lambda: "",  # Clear input box
        outputs=[msg]
    )
    
    msg.submit(
        fn=chat_interface,
        inputs=[msg, state, max_length_slider, temperature_slider],
        outputs=[chatbot, state],
    ).then(
        lambda: "",  # Clear input box
        outputs=[msg]
    )
    
    clear_btn.click(
        lambda: ([], []),
        outputs=[chatbot, state],
    )
    
    # Footer
    gr.Markdown("""
    ---
    ### About This Model
    - **Base Model:** TinyLlama-1.1B-Chat-v1.0
    - **Fine-tuned on:** Medical Q&A Dataset (33K+ samples)
    - **Training Method:** LoRA (Low-Rank Adaptation)
    - **Purpose:** Educational medical information assistant
    
    *Built with 🤗 Hugging Face Transformers & Gradio*
    """)

# Launch the app
if __name__ == "__main__":
    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        show_error=True,
    )
