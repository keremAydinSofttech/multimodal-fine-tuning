import os
import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
)
from peft.tuners.lora import LoraLayer
import pathlib
import wandb
import tensorflow as tf
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    tokenizer_image_token,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.constants import IMAGE_TOKEN_INDEX
from llava.utils import disable_torch_init
from llava import conversation as conversation_lib
from llava.model import *
from llava.train.train import (LLaVATrainer,
                               find_all_linear_names, 
                               get_peft_state_maybe_zero_3,
                               get_peft_state_non_lora_maybe_zero_3,
                               get_mm_adapter_state_maybe_zero_3,
                               smart_tokenizer_and_embedding_resize)
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from preprocessing  import *


class ModelTraining:

    def __init__(self):

        self.wandb_entity = 'bulentsiyah-softtech'
        self.wandb_project = 'llava_parameter_optimization'
    
    def wandb_init(self,):
        
        print("wandb_project",self.wandb_project)
        wandb.login()
        wandb.init(project=self.wandb_project, job_type="training", entity=self.wandb_entity, name=self.wandb_name)

        try:
            if len(self.wandb_project) > 0:
                os.environ["WANDB_PROJECT"] = self.wandb_project
        except:
            print("len(self.wandb_project) > 0")

    def make_supervised_data_module(self, data_args):
        """Make dataset and collator for supervised fine-tuning."""
        
        train_dataset = LazySupervisedDataset(tokenizer=self.tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args,
                                    image_processor=self.image_processor,
                                    image_folder=data_args.train_image_folder)

        eval_dataset = LazySupervisedDataset(tokenizer=self.tokenizer,
                                    data_path=data_args.eval_data_path,
                                    data_args=data_args,
                                    image_processor=self.image_processor,
                                    image_folder=data_args.eval_image_folder)
        
        
        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

        return dict(train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=data_collator)

    def setup_image(self, img_path):
        """
        Load and process the image.
        """

        if img_path.startswith('http') or img_path.startswith('https'):
            response = requests.get(img_path)
            conv_img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            conv_img = Image.open(img_path).convert('RGB')
        
        img_tensor = self.image_processor.preprocess(conv_img,  return_tensors='pt')['pixel_values']

        return img_tensor.to(dtype=self.model.dtype, device=self.model.device)

    def generate_answer(self, question, img_path):

        """
        Generates answer from fine-tuned model
        """
        
        '''
        conv_llava_v1 = Conversation(
                        system="A chat between a curious human and an artificial intelligence assistant. "
                            "The assistant gives helpful, detailed, and polite answers to the human's questions.",
                        roles=("USER", "ASSISTANT"),
                        version="v1",
                        messages=(),
                        offset=0,
                        sep_style=SeparatorStyle.TWO,
                        sep=" ",
                        sep2="</s>",
                    )
        '''
        
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        image_tensor = self.setup_image(img_path)

        # just one turn, always prepend image token
        inp = "<image>" + '\n' + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids=self.model.generate(
                                inputs=input_ids,
                                images=image_tensor,
                                do_sample=True,
                                temperature=0.01,
                                use_cache=True,
                                stopping_criteria=[stopping_criteria],
                                )

        return self.tokenizer.batch_decode(output_ids, 
                                     skip_special_tokens=True)[0].strip()

    def training_run(self, attn_implementation=None):

        disable_torch_init()

        # Parser extracts the parameters for arguments from run_script.sh
        parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args_filename='/mnt/bulentsiyah/multimodal-fine-tuning/llava/run_script_6.sh')

        self.wandb_name = '-'.join(training_args.output_dir.split('/')[-3:])
        self.wandb_init()
        
        compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        bnb_model_from_pretrained_args = {}
        if training_args.bits in [4, 8]:
            bnb_model_from_pretrained_args.update(dict(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
                )))

    
        # Initialization of the multimodal model
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            device_map='auto',
            **bnb_model_from_pretrained_args
            )
        self.model.config.use_cache = False

        # Stopping the training the backbone weights
        if model_args.freeze_backbone:
            self.model.model.requires_grad_(False)

        if training_args.bits in [4, 8]:
            self.model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
            #self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        if training_args.gradient_checkpointing:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # Initialization of Lora parameters
        if training_args.lora_enable:
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(self.model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type='CAUSAL_LM'
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    self.model.to(torch.bfloat16)
                if training_args.fp16:
                    self.model.to(torch.float16)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Initialization of tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            device_map='auto'
        )

        # Tokenizer pad token
        #self.tokenizer.pad_token = self.tokenizer.unk_token

        if model_args.version == "v0":
            if self.tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="[PAD]"),
                    tokenizer=self.tokenizer,
                    model=self.model,
                )
            elif model_args.version == "v0.5":
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                if model_args.version in conversation_lib.conv_templates:
                    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
                else:
                    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

        # Initialize the image processor
        self.model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=self.model.device)
        self.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        # Initialize the vision tokenizer
        self.model.config.image_aspect_ratio = data_args.image_aspect_ratio
        self.model.config.tokenizer_padding_side = self.tokenizer.padding_side
        self.model.config.tokenizer_model_max_length = self.tokenizer.model_max_length

        self.model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            self.model.requires_grad_(False)
            for p in self.model.get_model().mm_projector.parameters():
                p.requires_grad = True

        self.model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in self.model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            self.model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        self.model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        self.model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        self.model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        self.model.initialize_vision_tokenizer(model_args, tokenizer=self.tokenizer)

        if training_args.bits in [4, 8]:
            for name, module in self.model.named_modules():
                if isinstance(module, LoraLayer):
                    if training_args.bf16:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if training_args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

        # Data load
        data_module = self.make_supervised_data_module(data_args=data_args)

        # Trainer initialization
        trainer = LLaVATrainer(model=self.model,
                        tokenizer=self.tokenizer,
                        args=training_args,
                        **data_module)

        # Train the model
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()

        self.model.config.use_cache = True

        # Saving the model parameters
        state_dict = get_peft_state_maybe_zero_3(
            self.model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters()
        )

        print('Saving config of pretrained model...')
        self.model.config.save_pretrained(training_args.output_dir)

        print('Saving pretrained model...')
        self.model.save_pretrained(training_args.output_dir, state_dict=state_dict)

        print('Saving non lora trainables...')
        torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))

        keys_to_match = ['mm_projector']
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)

        print('Saving trainer model config...')
        trainer.model.config.save_pretrained(training_args.output_dir)

        print('Saving mm projector bin...')
        torch.save(weight_to_save, os.path.join(training_args.output_dir, 'mm_projector.bin'))

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

    attn_implementation = "flash_attention_2"

    print("----Training Started----")
    modelTraining = ModelTraining()
    training = modelTraining.training_run(attn_implementation)

    print("----Training ended----")