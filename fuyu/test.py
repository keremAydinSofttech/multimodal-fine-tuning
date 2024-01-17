import transformers
from peft import (
    PeftModel,
)
from PIL import Image
from configmanager import ConfigurationManager
from preprocessing import *
from constants import *

class ModelFuyu:

    def __init__(self):
        
        configurationManager = ConfigurationManager(config_file_path='./config.yaml')
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
        
        # Parser extracts the parameters for arguments from run_script.sh
        parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args_filename='run_script.sh')

        # Initialization of the multimodal model
        model_config = FuyuConfig()
        self.model = FuyuForCausalLM(model_config).from_pretrained('./8b_base_model_release/')
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path)

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
        inputs_to_model = self.processor(text=prompt, images=image_pil)

        generated_ids = self.model.generate(**inputs_to_model)
        
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
   

if __name__ == '__main__':

    print("----test basladı----")

    model_llava = ModelFuyu()
    
    prediction = model_llava.generate_answer(prompt= 'Why is the car stopping?', 
                                             img_path='C:/Users/201735/multimodal-fine-tuning/data/Screenshot.png')

    print(prediction)

    print("----test bitti----")



