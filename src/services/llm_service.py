from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.config.settings import settings

class LLMService:
    """Service for managing and interacting with Language Models."""
    
    def __init__(self, model_name: str = settings.DEFAULT_MODEL):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model(model_name)
    
    def load_model(self, model_name: str) -> None:
        """Load a specific LLM model."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        quantization_config = None
        
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        try:
            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        except Exception as e:
            print(f"Error loading model with device map {device_map}: {e}")
            print("Falling back to CPU only...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                trust_remote_code=True
            )
    
    def get_available_models(self) -> List[Dict]:
        """Return list of available models."""
        return settings.AVAILABLE_MODELS
    
    def generate_response(self, prompt: str, context: str = "", max_length: int = 1024) -> str:
        """Generate response using the loaded LLM."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized.")
        
        system_message = "You are a helpful assistant that answers questions about emails."
        full_prompt = f"Context from emails:\n{context}\n\nQuestion: {prompt}\n\nAnswer:" if context else f"{prompt}\n\nAnswer:"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response.split("Answer:")[-1].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}" 