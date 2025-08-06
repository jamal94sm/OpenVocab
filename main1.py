import torch
import transformers
import numpy as np
import random
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import MyDatasets
import MyModels
import MyPlayers
import MyUtils
import torchvision
import time
import json
import os
import gc
from sklearn.metrics import accuracy_score
from Config import args 
import time
import psutil


















def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)
    transformers.set_seed(seed)





def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


separator = "=" * 40



##################################################################################
##################################################################################
def main():

    
    device = torch.device(args.device)
    print(f'Device: {device}')




    # ===================== Client and Server Setup =====================
    p_Model = MyModels.Decoder_plus_FM(FM, processor, tokenizer, num_classes, name_classes).to(device)
    server = MyPlayers.Server(p_Model, None)


    
    server.train_decoder(num_classes)

    


    plt.plot(server.Loss)
    file_name = "A"
    plt.savefig(f"plots/{file_name}.png")
    










##################################################################################
##################################################################################
if __name__ == "__main__":
    
    
    set_seed(42)



    # ===================== Dataset and Model Loading =====================
    Dataset, num_classes, name_classes = MyDatasets.load_data_from_Huggingface()




    # ===================== Run for each configuration =====================
    configurations = [
        {"setup": "ft_M_yn_mean"},
    ]



    for config in configurations:
        args.setup = config["setup"]
        print(f"\n{separator} Running configuration: {args.setup} {separator}")
    
        
        ### Load the CLIP model for each setup 
        FM, processor, tokenizer = MyModels.load_clip_model()
        
        main()


        clean_memory()
        print(f"{separator} Simulation is over for configuration {args.setup} {separator}\n")

    


