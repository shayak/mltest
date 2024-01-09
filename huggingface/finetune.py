import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
import os,torch, wandb
from datasets import load_dataset
from trl import SFTTrainer

from huggingface import (
    defs,
    hfutils
)

MODEL_NUM = 2
DATASET_NUM = 0
FT_BASE_DIR = f'{hfutils.BASE_DIR}/{defs.models[MODEL_NUM]["name"]}/finetunes/{defs.datasets[DATASET_NUM]}'


def finetune(model, tokenizer, dataset):
    peft.prepare_model_for_kbit_training(model)
    peft_params = peft.LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_params = TrainingArguments(
        output_dir="./lora_results",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=400,
        logging_steps=200,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    torch.cuda.empty_cache()
    trainer.train()

    return trainer


def train_and_save():
    model, tokenizer = hfutils.load_local_model(
        defs.models[MODEL_NUM]['name'], defs.models[MODEL_NUM]['gptq']
    )
    dataset = load_dataset(defs.datasets[DATASET_NUM])

    # train
    trainer = finetune(model, tokenizer, dataset)

    print(FT_BASE_DIR)

    trainer.model.save_pretrained(FT_BASE_DIR)

    # trainer.model.push_to_hub(new_model)


def load_finetuned_model():
    base_model, tokenizer = hfutils.load_local_model(defs.models[MODEL_NUM]['name'], defs.models[MODEL_NUM]['gptq'])
    new_model = peft.PeftModel.from_pretrained(base_model, FT_BASE_DIR)
    del base_model
    torch.cuda.empty_cache()
    merged_model = new_model.merge_and_unload()
    # tokenizer = AutoTokenizer.from_pretrained(FT_BASE_DIR)
    return merged_model, tokenizer


def run():
    # train_and_save()
    model, tokenizer = load_finetuned_model()
    hfutils.chat_with_model(model, tokenizer)


if __name__ == "__main__":
    run()
