import torch
import tensorflow as tf



class CheckGPUAvailability:
    def __init__(self,):
        print("----------------------")
        print(torch.__version__)
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.device(0))
        print(torch.cuda.get_device_name(0))
        print("----------------------")
        print(tf.__version__)
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) 

if __name__ == '__main__':
    checkGPUAvailability = CheckGPUAvailability()