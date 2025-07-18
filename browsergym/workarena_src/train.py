# import os
# import time
# import json
# import warnings
# import torch

# import trl
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
import wandb
import pandas as pd
import pyarrow as pa
from datasets import Dataset
from tqdm import tqdm

import sys
sys.path.append('./')
from workarena_src.utils import load_hfmodel
#from hf_eval_rollout import run_eval_multiturn_repr, WebshopAgent
from evaluate import load
accuracy = load("accuracy")


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = logits.argmax(-1)
    return accuracy.compute(predictions=predictions, references=labels)

# import peft

#####################################################################
def main(cfg):
    # load dataset for training and validation
    train_df = pd.read_csv(f'/home/ubuntu/lcow-fork/browsergym/sft_train.csv')
    train_dataset = Dataset(pa.Table.from_pandas(train_df))
    val_df = pd.read_csv(f'/home/ubuntu/lcow-fork/browsergym/sft_train.csv')
    val_dataset = Dataset(pa.Table.from_pandas(val_df)) # Corrected: Use val_df for val_dataset

    # load model and tokenizers
    base_model, tokenizer = load_hfmodel()
    tokenizer.padding_side = 'right'

    training_args = SFTConfig(
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        output_dir=cfg.ckpt_dir,
        per_device_train_batch_size=cfg.per_gpu_bsz,
        per_device_eval_batch_size=cfg.per_gpu_bsz,
        fp16=False,
        bf16=True, # Keep BF16 for memory efficiency
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False}, # ADDED FOR DDP + GRADIENT CHECKPOINTING
        learning_rate=cfg.lr,
        logging_steps=cfg.logging_steps,
        num_train_epochs=cfg.n_epochs,
        warmup_ratio = cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        report_to="wandb",
        save_strategy="epoch",
        eval_on_start = True,
        # evaluation_strategy="epoch",
        seed=cfg.seed,
        group_by_length=True,
        max_seq_length=64, # MOVED HERE AND SET TO A REASONABLE VALUE
    )

    sep_tokens = tokenizer.encode('<|start_header_id|>assistant<|end_header_id|>')[1:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template = sep_tokens,
                                                    tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # dataset_text_field='text', # Uncomment if your CSV has a 'text' column for content
        # max_seq_length is now in SFTConfig, so remove it from here
        # tokenizer is typically inferred or not directly passed to SFTTrainer in newer trl versions
        args=training_args,
        data_collator = data_collator,
        compute_metrics=compute_metrics,
    )

    print('Set Trainer')
    print('Start Training!')
    trainer.train()
    

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Training contextualization module')
    
    # Add arguments
    parser.add_argument('--backbone', type=str, help='type of the action model backbone', default='googleai/gemini-1.5-flash-002')
    parser.add_argument('--iter', type=int, help='iteration for data collection', default=0)
    args = parser.parse_args()

    run = wandb.init(
        project="LCoW",
        name=f"Workarena-TinyLlama-1.1B-Chat-training",
    )

    # config
    cfg = wandb.config
    cfg.seed = 0
    cfg.model_name = "microsoft/Phi-3-mini-128k-instruct"
    cfg.ckpt_dir = f'ckpt/workarena_sft_iter_{args.iter}'
    cfg.per_gpu_bsz = 1
    cfg.gradient_accumulation_steps = 16  # Consider increasing if 2048 seq_len is still OOM
    cfg.lr = 1e-5
    cfg.logging_steps = 1
    cfg.n_epochs = 4
    cfg.weight_decay = 0.1
    cfg.warmup_ratio = 0.1
    cfg.eval_steps = 50
    cfg.save_steps = 100
    cfg.iter = args.iter
    cfg.backbone = args.backbone  # This might not be used if load_hfmodel uses cfg.model_name
    cfg.iter = args.iter  # Duplicate, can remove one

    main(cfg)




    
