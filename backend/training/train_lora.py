"""Small LoRA fine-tuning starter. Use a GPU and review the model license first.
Dataset: JSONL, one object per line, e.g. {"text":"### User: Hi\n### Assistant: Hello!"}
Run from backend/: python -m venv .venv ... then pip install -r training-requirements.txt
"""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='training/data.jsonl'); p.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct'); p.add_argument('--out',default='training/output'); p.add_argument('--epochs',type=float,default=1); p.add_argument('--max-length',type=int,default=512); args=p.parse_args()
    ds=load_dataset('json', data_files=args.data, split='train')
    if 'text' not in ds.column_names: raise ValueError('Each JSONL row must contain a text field.')
    tok=AutoTokenizer.from_pretrained(args.model); tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model)
    model=get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj','v_proj']))
    def tokenize(batch): return tok(batch['text'], truncation=True, max_length=args.max_length)
    tokenized=ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    training=TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=5, save_strategy='epoch', fp16=False, report_to='none')
    trainer=Trainer(model=model, args=training, train_dataset=tokenized, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train(); model.save_pretrained(args.out); tok.save_pretrained(args.out); print(f'LoRA adapter saved to {args.out}. Merge/export separately if needed.')
if __name__=='__main__': main()
