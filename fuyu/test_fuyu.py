import transformers
from peft import (
    PeftModel,
)
from PIL import Image
from preprocessing import *
from constants import *
from transformers import FuyuForCausalLM
from transformers.models.fuyu.processing_fuyu import FuyuProcessor
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor
from transformers.models.fuyu.configuration_fuyu import FuyuConfig
import warnings
warnings.filterwarnings("ignore")

class ModelFuyu:

    def __init__(self):
        
        self.experiments_number = '1'
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.load_models()

    def load_models(self):
        
        '''
        Load the model, process and tokenizer
        '''
        
        # Parser extracts the parameters for arguments from run_script.sh
        parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args_filename='run_script.sh')

        # Initialization of the multimodal model
        model_config = FuyuConfig()
        self.model = FuyuForCausalLM(model_config).from_pretrained(model_args.model_name_or_path,
                                                                   device_map='auto')
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                device_map='auto')

        # Initialize the image processor
        image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(image_processor=image_processor, tokenizer=self.tokenizer)

        #self.model=PeftModel.from_pretrained(self.model, self.base_path)  

        #self.model.to(training_args.device, dtype=torch.float16)      

    def generate_answer(self, prompt, img_path):

        """
        Generates answer from fine-tuned model
        """

        image_pil = Image.open(img_path).convert('RGB')
        inputs_to_model = self.processor(text=prompt, images=image_pil, device_map=self.model.device).to(device=self.model.device)

        generated_ids = self.model.generate(**inputs_to_model, max_new_tokens=20, pad_token_id=50256)
        
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split('<s>')[-1]
   

if __name__ == '__main__':

    print("----test basladı----")

    model_llava = ModelFuyu()
    
    prediction = model_llava.generate_answer(prompt= "Should I stop when I'm about to cross the street?", 
                                             img_path='../data/individual_evaluation/commonse_reasoning.png')

    print(prediction)

    print("----test bitti----")



