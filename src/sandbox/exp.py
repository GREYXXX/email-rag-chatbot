# Install required packages if not already installed
# !pip install transformers accelerate torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# List of LLMs to evaluate
llms = [
    {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "size": "1.1B", "description": "Small but efficient chat model"},
    {"name": "microsoft/phi-1_5", "size": "1.3B", "description": "Microsoft's Phi-1.5 small model"},
    {"name": "deepseek-ai/deepseek-coder-1.3b-base", "size": "1.3B", "description": "Code-focused small model"},
    {"name": "databricks/dolly-v2-3b", "size": "3B", "description": "Dolly instruction model (slightly larger)"},
    {"name": "meta-llama/Llama-2-7b-chat-hf", "size": "7B", "description": "Llama 2 chat model (requires approval)"}
]

# Questions to ask each model
questions = [
    "What are the main use cases for small language models?",
    "Can you explain the differences between instruction-tuned and base models?",
    "Give a simple code example using a Python dictionary.",
    "Why is model size important for inference on edge devices?",
]

# Load and chat with each model
def chat_with_models(llms, questions):
    for model in llms:
        model_name = model["name"]
        print(f"\n{'='*60}\n🔍 Testing: {model_name} ({model['description']})\n{'='*60}")

        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_instance = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            pipe = pipeline("text-generation", model=model_instance, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

            # Ask each question
            for q in questions:
                print(f"\n❓ Question: {q}")
                prompt = q + "\n\nAnswer:"
                response = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)[0]['generated_text']
                answer = response[len(prompt):].strip()
                print(f"💬 Response: {answer}")

        except Exception as e:
            print(f"⚠️ Error loading {model_name}: {e}")

if __name__ == "__main__":
    chat_with_models(llms, questions)
