from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import os
import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
)
import pathlib
import wandb
import tensorflow as tf
from configmanager import ConfigurationManager
from llava.model import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from llava import conversation as conversation_lib
from llava.model import *
from llava.train.train import (LLaVATrainer, 
                               DataArguments, 
                               ModelArguments, TrainingArguments, 
                               DataCollatorForSupervisedDataset, 
                               find_all_linear_names, 
                               get_peft_state_maybe_zero_3,
                               get_peft_state_non_lora_maybe_zero_3)
from preprocessing  import *

class ModelTraining:

    def __init__(self):

        self.configurationManager = ConfigurationManager(config_file_path='./config.yaml')
        self.experiments_number = self.configurationManager.config_readable['experiments_number']
        self.epoch = int(self.configurationManager.config_readable['epoch'])
        self.wandb_entity = self.configurationManager.config_readable['wandb_entity']
        self.translation_combined_full_data = self.configurationManager.config_readable['translation_combined_full_train']
        self.base_path = "./fine_tuning_llava_beta/"+self.experiments_number
        
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv = None

        self.wandb_project = "llava-beta_"+self.experiments_number+"_epoch_"+str(self.epoch)
    
    def wandb_init(self,):
        
        print("wandb_project",self.wandb_project)
        wandb.login()
        wandb.init(project=self.wandb_project, job_type="training", entity=self.wandb_entity)

        try:
            if len(self.wandb_project) > 0:
                os.environ["WANDB_PROJECT"] = self.wandb_project
        except:
            print("len(self.wandb_project) > 0")

    def make_supervised_data_module(self, tokenizer: transformers.PreTrainedTokenizer,
                                data_args):
        """Make dataset and collator for supervised fine-tuning."""
        
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args,
                                    image_processor=data_args.image_processor)
        
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        return dict(train_dataset=train_dataset,
                    eval_dataset=None,
                    data_collator=data_collator)

    def training_run(self):

        disable_torch_init()

        # Parser extracts the parameters for arguments from run_script.sh
        parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args_filename='run_script.sh')

        # Initialization of the multimodal model
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            )
        model.config.use_cache = False

        # Stopping the training the backbone weights
        if model_args.freeze_backbone:
            model.model.requires_grad_(False)

        if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # Initialization of Lora parameters
        if training_args.lora_enable:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=find_all_linear_names(model),
                lora_dropout=0.05,
                bias='none',
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            model = get_peft_model(model, lora_config)

        # Initialization of tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        # Tokenizer token
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

        # Initialize the image processor
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        vision_tower = model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        # Initialize the vision tokenizer
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        # Data load
        data_module = self.make_supervised_data_module(tokenizer=tokenizer,
                                                       data_args=data_args)

        # Trainer initialization
        trainer = LLaVATrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        **data_module)

        # Train the model
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()

        model.config.use_cache = True

        # Saving the model parameters
        if training_args.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )

            trainer.save_model(training_args.output_dir)

            tokenizer.save_pretrained(training_args.output_dir)

            if training_args.local_rank == 0 or training_args.local_rank == -1:
                model.config.save_pretrained(training_args.output_dir)
                model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))


class CheckGPUAvailability:
    def __init__(self,):
        print("--------torch--------------")
        print(torch.__version__)
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.device(0))
        print(torch.cuda.get_device_name(0))
        print("----------------------")
        print(tf.__version__)
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) 
        print("--------tf--------------")



if __name__ == '__main__':
    checkGPUAvailability = CheckGPUAvailability()

    replace_llama_attn_with_flash_attn()

    print("----Training Started----")
    modelTraining = ModelTraining()
    training = modelTraining.training_run()

    print("----Training ended----")