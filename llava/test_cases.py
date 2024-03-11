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
import pandas as pd
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
        
        # Parser extracts the parameters for arguments from run_script.sh
        parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
        model_args, _, training_args = parser.parse_args_into_dataclasses(args_filename='/mnt/bulentsiyah/multimodal-fine-tuning/llava/run_script_10.sh')

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

    def load_trained_model(self, base_path):

        self.trained_model=PeftModel.from_pretrained(self.model, base_path)  

        self.trained_model.to(self.model.device, dtype=torch.float16)  
        

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

    print("----test basladı----")

    with open('/mnt/bulentsiyah/multimodal-fine-tuning/data/jsons/df_general_test.json', 'r') as f:
        test_data = json.load(f)

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    image_folder = '/mnt/bulentsiyah/multimodal-fine-tuning/data/datasets/syn_dataset/val_data/'

    model_llava = ModelLLaVa()

    df = pd.read_csv('results.csv')

    scores = {}    

    for data_type in os.listdir('/mnt/bulentsiyah/multimodal-fine-tuning/llava/results/parameter_optimization'):

        folder_path = os.path.join('/mnt/bulentsiyah/multimodal-fine-tuning/llava/results/parameter_optimization', data_type)

        if f'{data_type}' in df.columns:
            continue

        # Check if any filename matches the pattern
        try:
            if not 'adapter_config.json' in os.listdir(folder_path):
                continue
        except:
            continue

        scores[f'{data_type}'] = {'hour':0, 'minute':0}

        torch.cuda.empty_cache()
    
        model_llava.load_trained_model(base_path=folder_path)

        for i in range(len(test_data)):

            print(f'{round((i+1) / (len(test_data)) * 100, 2)}%')

            prediction = model_llava.generate_answer(prompt= test_data[i]['conversations'][0]['value'].split('<image>\n')[-1], 
                                                img_path=os.path.join(image_folder, test_data[i]['image']))

            answer = test_data[i]['conversations'][1]['value']

            #print('answer:', answer)
            #print('prediction:', prediction)

            if prediction.find(f'{answer.split(":")[0]}:') != -1:
                scores[f'{data_type}']['hour'] += 1
            if prediction.find(f':{answer.split(":")[1]}') != -1:
                scores[f'{data_type}']['minute'] += 1

        df_scores = pd.DataFrame([scores])

        df_scores.to_csv('results3.csv', index=False)
    
    print("----test bitti----")



