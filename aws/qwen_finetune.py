"""
Qwen 2.5 7B QLoRA Fine-Tuning Pipeline for Entity Resolution
Optimized for AWS EC2 g5.xlarge (1x NVIDIA A10G 24GB GPU)
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Base model on Hugging Face (Apache 2.0 license)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def run_qwen_finetuning(
    data_path: str = "aws/data/train_pairs_sample.jsonl",
    output_dir: str = "aws/models/qwen_er_adapter",
    num_epochs: int = 2,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4
):
    print("=" * 60)
    print(f"STARTING QWEN 2.5 7B QLoRA FINE-TUNING")
    print(f"• Base Model: {MODEL_ID}")
    print(f"• Dataset: {data_path}")
    print(f"• Output Directory: {output_dir}")
    print(f"• CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"• GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"• Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60)

    # 1. 4-bit Quantization Config (NormalFloat4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load Model in 4-bit
    print("Loading base model in 4-bit NF4...")
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    )

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA Adapter Config (targeting all linear attention & MLP projections)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    model = get_peft_model(model, peft_config)
    print("\nTrainable parameters summary:")
    model.print_trainable_parameters()

    # 5. Load Dataset
    print(f"\nLoading training dataset from {data_path}...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Total examples loaded: {len(dataset):,}")

    # Formatting function using Qwen's ChatML template
    def formatting_prompts_func(example):
        formatted_texts = []
        for msgs in example["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
        return formatted_texts

    # 6. Training Arguments
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none"
    )

    # 7. SFT Trainer
    print("\nInitializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="messages",
        formatting_func=formatting_prompts_func,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )

    print("\nStarting Training on GPU...")
    trainer.train()

    # 8. Save Final LoRA Adapter
    print(f"\nTraining completed! Saving LoRA adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("LoRA Adapter saved successfully (~100 MB). Ready for inference!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="aws/data/train_pairs_sample.jsonl")
    parser.add_argument("--output", default="aws/models/qwen_er_adapter")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    run_qwen_finetuning(
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
