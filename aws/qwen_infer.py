"""
Fast Borderline Case Inference using Fine-Tuned Qwen 2.5 7B LoRA Adapter
"""

import os
import sys
import json
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class QwenERJudge:
    def __init__(self, adapter_path: str = "aws/models/qwen_er_adapter"):
        print(f"Loading Qwen ER Judge with adapter from {adapter_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        
        # Load base model + LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        print("Qwen ER Judge initialized and ready for inference!")

    def evaluate_pair(self, rec_a: dict, rec_b: dict) -> Dict:
        """Evaluates a single pair and returns parsed decision dictionary."""
        system_msg = (
            "You are an expert Entity Resolution auditor. "
            "Analyze Record A and Record B and determine if they refer to the exact same real-world business entity. "
            "Return a valid JSON object with keys: 'is_match' (boolean), 'confidence' (float 0.0-1.0), and 'rationale' (string)."
        )
        user_msg = (
            f"Record A:\nName: {rec_a.get('business_name','')}\nAddress: {rec_a.get('business_address','')}\nCountry: {rec_a.get('country','')}\n\n"
            f"Record B:\nName: {rec_b.get('business_name','')}\nAddress: {rec_b.get('business_address','')}\nCountry: {rec_b.get('country','')}"
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        try:
            return json.loads(response_text)
        except Exception:
            # Fallback parsing
            is_match = "true" in response_text.lower()
            return {"is_match": is_match, "confidence": 0.85 if is_match else 0.15, "raw": response_text}
