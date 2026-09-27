"""
VIRTUS-ML: High-Speed Batch Inference with Fine-Tuned Qwen 2.5 7B QLoRA Arbiter
Evaluates ambiguous borderline entity resolution pairs on GPU.
"""

import os
import sys
import json
import gzip
import time
import argparse
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are an international business entity resolution expert fluent in multiple languages including English, Hindi, and French. "
    "Analyze Record A and Record B and determine if they refer to the exact same real-world business entity despite differences in language, script, address formatting, abbreviations, or legal entity suffixes. "
    "Return a valid JSON object with keys: 'is_match' (boolean), 'confidence' (float 0.0-1.0), and 'rationale' (string)."
)


def load_model_and_tokenizer(adapter_path: str = "aws/models/qwen_er_adapter"):
    print("=" * 70)
    print(f"LOADING QWEN 2.5 7B ARBITER")
    print(f"• Base Model   : {BASE_MODEL_ID}")
    print(f"• Adapter Path : {adapter_path}")
    print(f"• CUDA Active  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"• GPU Device   : {torch.cuda.get_device_name(0)}")
        print(f"• Total VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if os.path.exists(adapter_path) else BASE_MODEL_ID,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("• Loading base model weights in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    )

    if os.path.exists(adapter_path):
        print(f"• Merging fine-tuned LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("! WARNING: Adapter path not found. Running with base Qwen-2.5-7B-Instruct zero-shot.")
        model = base_model

    model.eval()
    print("✓ Model and Tokenizer successfully ready!")
    return model, tokenizer


def parse_llm_json(raw_text: str) -> dict:
    """Extract and parse json from LLM output."""
    raw = raw_text.strip()
    # Try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    
    # Try finding first { and last }
    s = raw.find('{')
    e = raw.rfind('}')
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(raw[s:e+1])
        except Exception:
            pass

    # Heuristic fallback
    lower = raw.lower()
    is_match = ("\"is_match\": true" in lower) or ("\"is_match\":true" in lower) or ("is_match = true" in lower)
    conf = 0.90 if is_match else 0.10
    return {"is_match": is_match, "confidence": conf, "rationale": raw[:80]}


def run_borderline_inference(
    input_file: str,
    output_file: str,
    adapter_path: str = "aws/models/qwen_er_adapter",
    batch_size: int = 16,
    max_samples: int = None
):
    model, tokenizer = load_model_and_tokenizer(adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Read input pairs
    print(f"\n[1/3] Reading borderline pairs from {input_file}...")
    opener = gzip.open if input_file.endswith('.gz') else open
    pairs = []
    with opener(input_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
                if max_samples and len(pairs) >= max_samples:
                    break

    total = len(pairs)
    print(f"• Total pairs loaded: {total:,}")

    # 2. Batch Inference
    print(f"\n[2/3] Running GPU batch inference (Batch Size: {batch_size})...")
    decisions = []
    t0 = time.time()
    matches_count = 0
    rejects_count = 0

    for i in range(0, total, batch_size):
        batch = pairs[i:i + batch_size]
        prompts = []
        for p in batch:
            rec_a = p.get('record_a', {})
            rec_b = p.get('record_b', {})
            user_msg = (
                f"Record A:\n"
                f"Name: {rec_a.get('business_name', '')}\n"
                f"Address: {rec_a.get('business_address', '')}\n"
                f"Country: {rec_a.get('country', '')}\n\n"
                f"Record B:\n"
                f"Name: {rec_b.get('business_name', '')}\n"
                f"Address: {rec_b.get('business_address', '')}\n"
                f"Country: {rec_b.get('country', '')}"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.01,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        input_len = inputs.input_ids.shape[1]
        for idx, p in enumerate(batch):
            gen_tokens = outputs[idx][input_len:]
            resp_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            parsed = parse_llm_json(resp_text)

            is_match = parsed.get('is_match', False)
            conf = float(parsed.get('confidence', 0.5))

            if is_match and conf >= 0.70:
                matches_count += 1
            else:
                rejects_count += 1

            decisions.append({
                "source1_id": p.get('source1_id'),
                "candidate_id": p.get('candidate_id'),
                "is_match": is_match,
                "confidence": conf,
                "rationale": parsed.get('rationale', '')
            })

        processed = min(i + batch_size, total)
        elapsed = time.time() - t0
        speed = processed / max(elapsed, 0.001)
        eta_sec = (total - processed) / max(speed, 0.001)
        if (processed % (batch_size * 5) == 0) or processed == total:
            print(f"  • Progress: {processed:,}/{total:,} ({processed/total*100:.1f}%) | Speed: {speed:.1f} pairs/sec | ETA: {eta_sec/60:.1f} min | Confirmed Matches: {matches_count:,} | Rejections: {rejects_count:,}")

    # 3. Save Decisions
    print(f"\n[3/3] Saving {len(decisions):,} decisions to {output_file}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(decisions, f, indent=2)

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE:")
    print(f"• Total Processed   : {len(decisions):,} pairs")
    print(f"• Confirmed Matches : {matches_count:,}")
    print(f"• Rejected Matches  : {rejects_count:,}")
    print(f"• Time Elapsed      : {total_time:.1f}s ({total_time/60:.2f} mins)")
    print(f"• Output File       : {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen 2.5 7B borderline inference.")
    parser.add_argument("--input", default="borderline_cases_for_qwen.jsonl.gz", help="Path to input borderline cases")
    parser.add_argument("--output", default="qwen_decisions.json", help="Path to output decisions JSON")
    parser.add_argument("--adapter", default="aws/models/qwen_er_adapter", help="Path to LoRA adapter")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick test")
    args = parser.parse_args()

    run_borderline_inference(
        input_file=args.input,
        output_file=args.output,
        adapter_path=args.adapter,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
