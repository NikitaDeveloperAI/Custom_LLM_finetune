import sys
from tqdm import tqdm

import torch
from datasets import load_dataset
from random import randint
from transformers import TrainingArguments,  AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import pipeline
from trl import setup_chat_format


sys.path = ['C:\\Users\\NSKZ\\PycharmProjects\\PersonalLLM\\'] + sys.path


# Convert dataset to OAI messages
system_message = """You are an text to SQL query translator. 
Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""

def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": system_message.format(schema=sample["context"])},
      {"role": "user", "content": sample["question"]},
      {"role": "assistant", "content": sample["answer"]}
    ]
  }

# Load dataset from the hub
dataset = load_dataset("b-mc2/sql-create-context", split="train")
dataset = dataset.shuffle().select(range(12500))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
# split dataset into 10,000 training samples and 2,500 test samples
dataset = dataset.train_test_split(test_size=2500/12500)

# save datasets to disk
dataset["train"].to_json("train_dataset.json", orient="records")
dataset["test"].to_json("test_dataset.json", orient="records")


# Load jsonl data from disk
train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
test_dataset = load_dataset("json", data_files="test_dataset.json", split="train")

# Hugging Face model id
model_id = "codellama/CodeLlama-7b-hf"  # or `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or `mistralai/Mistral-7B-v0.1`

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'  # to prevent warnings

# # set chat template to OAI chatML, remove if you start from a fine-tuned model
model, tokenizer = setup_chat_format(model, tokenizer)


args = TrainingArguments(
    output_dir="llm-training",              # directory to save and repository id
    num_train_epochs=1,                     # number of training epochs
    max_steps=50,
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=6,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=5,                        # log every n steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                      # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)

max_seq_length = 3072  # max sequence length for model and packing of the dataset


peft_config = LoraConfig(
    r=64,  # the rank of the LoRA matrices
    lora_alpha=16,  # the weight
    lora_dropout=0.1,  # dropout to add to the LoRA layers
    bias="none",  # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj","v_proj","o_proj"],  # the name of the layers to add LoRA
    modules_to_save=None,  # layers to unfreeze and train from the original pre-trained model
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

trainer.train()

# save model
trainer.save_model()


model = AutoPeftModelForCausalLM.from_pretrained('llm-training', torch_dtype=torch.float16)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# Load our test dataset
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
rand_idx = randint(0, len(eval_dataset))

# Test on sample
prompt = pipe.tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt,
               max_new_tokens=256,
               do_sample=True,
               temperature=0.1,
               top_k=50,
               top_p=0.1,
               eos_token_id=pipe.tokenizer.eos_token_id,
               pad_token_id=pipe.tokenizer.pad_token_id,
               )

print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")

def evaluate(sample):
    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
    if predicted_answer == sample["messages"][2]["content"]:
        return 1
    else:
        return 0


success_rate = []
number_of_eval_samples = 10
# iterate over eval dataset and predict
for s in tqdm(eval_dataset.shuffle().select(range(number_of_eval_samples))):
    success_rate.append(evaluate(s))

# compute accuracy
accuracy = sum(success_rate)/len(success_rate)

print(f"Accuracy: {accuracy*100:.2f}%")

