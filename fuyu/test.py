import torch
import transformers
from peft import (
    PeftModel,
)
from transformers import (
    BitsAndBytesConfig
)
from PIL import Image
from io import BytesIO
import requests
from configmanager import ConfigurationManager


class ModelFuyu:

    def __init__(self):
        
        configurationManager = ConfigurationManager(config_file_path='./test_config_file.yaml')
        self.experiments_number = configurationManager.config_readable['experiments_number']
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv = None
        self.conv_img = None
        self.img_tensor = None
        self.roles = None
        self.stop_key = None
        self.load_models()

    def load_models(self):
        
        '''
        Load the model, process and tokenizer
        '''

        disable_torch_init()

        self.base_path = "./fine_tuning_llava_beta/" + self.experiments_number
        
        # Parser extracts the parameters for arguments from run_script.sh
        parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args_filename='run_script.sh')

        # The data type
        compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

        # Config initialization
        bnb_model_from_pretrained_args = {}
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
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
            )
        ))

        self.model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
        self.model.config.use_cache = True
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.base_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
            )

        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        self.image_processor = vision_tower.image_processor
        
        self.model=PeftModel.from_pretrained(self.model, self.base_path)  

        self.model.to(training_args.device, dtype=torch.float16)      

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

        return img_tensor.to(self.model.device, dtype=torch.float16)

    def generate_answer(self, prompt, img_path):

        """
        Generates answer from fine-tuned model
        """
        
        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()

        image_tensor = self.setup_image(img_path)

        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
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
                                stopping_criteria=[stopping_criteria],)

        return self.tokenizer.decode(output_ids[0, input_ids.shape[1] :], 
                                     skip_special_tokens=True).strip()
   

if __name__ == '__main__':

    print("----test basladı----")

    model_llava = ModelFuyu()
    
    prediction = model_llava.generate_answer(prompt= 'Why is the car stopping?', 
                                             img_path='Screenshot.png')

    print(prediction)

    print("----test bitti----")



