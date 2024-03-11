import torch
import transformers
from peft import (
    PeftModel,
)
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    tokenizer_image_token,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from PIL import Image
from io import BytesIO
from rouge_score import rouge_scorer
import requests
from llava.utils import disable_torch_init
from preprocessing import *

import warnings
warnings.filterwarnings("ignore")

class ModelLLaVa:

    def __init__(self):
        
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

        self.base_path = "/mnt/bulentsiyah/multimodal-fine-tuning/llava/fine_tuning_llava_beta_digital/checkpoint-500"
        
        # Parser extracts the parameters for arguments from run_script.sh
        parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
        model_args, _, training_args = parser.parse_args_into_dataclasses(args_filename='/mnt/bulentsiyah/multimodal-fine-tuning/llava/run_script.sh')

        self.model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    device_map='auto'
                )
        self.model.config.use_cache = True
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                device_map='auto'
            )

        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(device=self.model.device)
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

        return img_tensor.to(dtype=self.model.dtype, device=self.model.device)

    def generate_answer(self, prompt, img_path):

        """
        Generates answer from fine-tuned model
        """
        
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        image_tensor = self.setup_image(img_path)
        image_sizes = [x.size for x in image_tensor]

        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=self.model.device)

        with torch.inference_mode():
            output_ids=self.model.generate(
                                inputs=input_ids,
                                images=image_tensor,
                                image_sizes=image_sizes,
                                do_sample=True,
                                temperature=0.01,
                                use_cache=True,
                                )
            
        return self.tokenizer.decode(output_ids[0, input_ids.shape[0]:], 
                                     skip_special_tokens=True).strip()
   

if __name__ == '__main__':

    torch.cuda.empty_cache()

    print("----test basladı----")

    with open('/mnt/bulentsiyah/multimodal-fine-tuning/data/digital_clock_smaller.json', 'r') as f:
        test_data = json.load(f)

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    image_folder = '/mnt/bulentsiyah/multimodal-fine-tuning/data/digital_clock_test'

    final_score = 0

    model_llava = ModelLLaVa()

    one_guess = False

    if one_guess:

        image_path = '/mnt/bulentsiyah/multimodal-fine-tuning/data/6.png'

        prediction = model_llava.generate_answer(prompt= 'What time is it? Explain your reasoning.', 
                                                img_path=image_path)

        print(image_path)
        print('question: ', 'What time is it on the clock?')
        print('prediction:', prediction)

    else:

        for i in range(len(test_data)):

            print(f'{round((i+1) / (len(test_data)) * 100, 2)}%')

            prediction = model_llava.generate_answer(prompt= test_data[i]['conversations'][0]['value'].split('<image>\n')[-1], 
                                                img_path=os.path.join(image_folder, test_data[i]['image']))

            print(os.path.join(image_folder, test_data[i]['image']))
            print('question: ', test_data[i]['conversations'][0]['value'].split('<image>\n')[-1])
            print('answer:', test_data[i]['conversations'][1]['value'])
            print('prediction:', prediction)

            if prediction.find(test_data[i]['conversations'][1]['value'].split()[1].replace('.','')) != -1:
                rouge_score = 1
            else:
                rouge_score = 0

            final_score += rouge_score

        print('Test Sonucu', f'{round(final_score/len(test_data) * 100, 2)}%')
    
    print("----test bitti----")



