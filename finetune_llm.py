from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from datasets import load_dataset

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("formatting data")
dataset = load_dataset("json", data_files="data/examples.jsonl")["train"]

print("configurations")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,   
    device_map="auto",  
    dtype=torch.bfloat16,
    trust_remote_code=True             
)
print(next(model.parameters()).device)

# model.config.use_cache = False
# model.config.pretraining_tp = 1

from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    lora_alpha=16,                           # Scaling factor for LoRA
    lora_dropout=0.05,                       # Add slight dropout for regularization
    r=8,                                    # Rank of the LoRA update matrices
    task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # Target modules for LoRA
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("training setup")
from trl import SFTTrainer
from transformers import TrainingArguments

# Training Arguments
training_arguments = TrainingArguments(
    output_dir="models/finetune-style",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    logging_steps=10,
    learning_rate=1e-4,
    fp16=False,
    bf16=False,
    save_strategy="epoch",
    report_to="none"
)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)

print("training")
trainer.train()

trainer.model.save_pretrained("./models/finetune-style-lora")
tokenizer.save_pretrained("./models/finetune-style-lora")
