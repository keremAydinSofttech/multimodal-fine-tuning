import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb
import os

import argparse
import torch
import tensorflow as tf

from configmanager import ConfigurationManager

class ModelTesting:
    """
    Testin yapildigi siniftir.
    """

    def __init__(self,  video_id="none_name"):
        self.configurationManager = ConfigurationManager()
        self.experiments_number = self.configurationManager.config_readable['experiments_number']
        self.code_language = self.configurationManager.config_readable['code_language']
        self.epoch = int(self.configurationManager.config_readable['epoch'])
        self.wandb_entity = self.configurationManager.config_readable['wandb_entity']
        self.java_combined_full_train = self.configurationManager.config_readable['java_combined_full_train']
        self.cbl_combined_full_train = self.configurationManager.config_readable['cbl_combined_full_train']
        self.java_combined_full_test = self.configurationManager.config_readable['java_combined_full_test']
        self.cbl_combined_full_test = self.configurationManager.config_readable['cbl_combined_full_test']

        self.tokenizer = None
        self.model = None

        self.base_path = "./fine_tuning_starcoder_chat_beta/"+self.experiments_number

        self.wandb_project = "starchat-beta_"+self.code_language+"_"+self.experiments_number+"_epoch_"+str(self.epoch)

        self.train_data = None
        self.test_data = None


    def model_init(self):
        self.wandb_init()
        model_name = "HuggingFaceH4/starchat-beta"
        model_tok = "HuggingFaceH4/starchat-beta"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,#false memory kazanmak ıcın
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_tok)
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token = "<fim_pad>"
        self.tokenizer.eos_token = "<|endoftext|>"

        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        falcon_lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query_key_value"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

    
        args = argparse.ArgumentParser()
        args.lora_r = 16
        args.lora_alpha = 32
        args.lora_dropout = 0.05

        starcoder_lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules = ["c_proj", "c_attn", "q_attn"]
        )

        self.model = get_peft_model(self.model, starcoder_lora_config)
        self.print_trainable_parameters(self.model)

        return True


    def wandb_init(self,):
        
        print("wandb_project",self.wandb_project)
        wandb.login()
        wandb.init(project=self.wandb_project, job_type="training", entity=self.wandb_entity)

        try:
            if len(self.wandb_project) > 0:
                os.environ["WANDB_PROJECT"] = self.wandb_project
        except:
            print("len(self.wandb_project) > 0")



    def print_trainable_parameters(self,model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
        )


    def load_dataset(self,):
        #train_data_merged = load_dataset("csv", data_files=[self.java_combined_full_train,self.cbl_combined_full_train])
        train_data_merged = load_dataset("csv", data_files=[self.java_combined_full_train])
        print("train_data_merged",train_data_merged)

        #test_data_merged= load_dataset('csv', data_files={'test': [self.java_combined_full_test,self.cbl_combined_full_test]})
        test_data_merged= load_dataset('csv', data_files={'test': [self.java_combined_full_test]})
        print("test_data_merged",test_data_merged)

        self.train_data = train_data_merged["train"].shuffle().map(self.generate_and_tokenize_prompt)
        self.test_data = test_data_merged["test"].shuffle().map(self.generate_and_tokenize_prompt)

        return True

        

    def generate_prompt(self,data_point):
            return f"""
            : {data_point["prompt"]}
            : {data_point["question"]}
            <|endoftext|>
            """.strip()


    def generate_and_tokenize_prompt(self,data_point):
        full_prompt = self.generate_prompt(data_point)
        tokenized_full_prompt = self.tokenizer(full_prompt, padding=True, truncation=True)
        return tokenized_full_prompt


    def testing_run(self,):
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            num_train_epochs=self.epoch,
            learning_rate=2e-4,
            fp16=True,
            save_total_limit=3,
            logging_steps=1,
            output_dir=self.base_path,
            optim="adamw_torch", #adamw_hf, #paged_adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            #yeniler
            run_name=self.wandb_project,
            report_to="wandb",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=20,
            save_steps=50,
        )

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.train_data,
            eval_dataset = self.test_data,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        self.model.config.use_cache = False

        peft_starcoder_chat_save_pretrained = self.base_path+"/save_pretrained"
        os.mkdir(peft_starcoder_chat_save_pretrained) 
        print("peft_starcoder_chat_save_pretrained '%s' created" %peft_starcoder_chat_save_pretrained) 
        trainer.train()

        
        self.model.save_pretrained(peft_starcoder_chat_save_pretrained)

        wandb.finish()

        peft_starcoder_chat_trainer_save_model = self.base_path+"/trainer_save_model"
        os.mkdir(peft_starcoder_chat_trainer_save_model) 
        print("peft_starcoder_chat_trainer_save_model '%s' created" %peft_starcoder_chat_trainer_save_model) 
        trainer.save_model(peft_starcoder_chat_trainer_save_model)

        peft_starcoder_chat_tokenizer_save_model = self.base_path+"/tokenizer_save_model"
        os.mkdir(peft_starcoder_chat_tokenizer_save_model) 
        print("peft_starcoder_chat_tokenizer_save_model '%s' created" %peft_starcoder_chat_tokenizer_save_model) 
        self.tokenizer.save_pretrained(peft_starcoder_chat_tokenizer_save_model)




if __name__ == '__main__':
    print("----test basladı----")
    model_init = ModelTesting.model_init()
    if model_init:
        load_dataset = ModelTesting.load_dataset()
        if load_dataset:
            testing_run = ModelTesting.testing_run()

    print("----test bitti----")



