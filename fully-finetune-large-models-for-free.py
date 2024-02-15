#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().system('cp /kaggle/input/utils-xla/spmd_util.py . # From this repo: https://github.com/HeegyuKim/torch-xla-SPMD')


# In[ ]:


import os
import pandas as pd
import numpy as np
import datasets
import torch.optim as optim
import torch_xla.debug.profiler as xp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp # We also import mp modules if we wanna use that for some reason
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
import torch
import torch.nn as nn
import re
import torch_xla.experimental.xla_sharding as xs
import torch_xla.core.xla_model as xm
from transformers import (
    GPTNeoXConfig, T5Config, LlamaConfig, AutoTokenizer, AutoModelForCausalLM, MistralConfig, DataCollatorWithPadding, AutoConfig, AutoModelForSequenceClassification
) # You can use any of models with those configs (even flan T5 xxl!). Other models are not supported.

from transformers import logging as hf_logging
import torch.nn.functional as F
import torch_xla.runtime as xr

xr.use_spmd()

import torch_xla.experimental.xla_sharding as xs # "experimental" prefix always means you're gonna have a good time LMAO
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.xla_sharding import Mesh

from peft import LoraConfig, TaskType, get_peft_model # If we wanna use peft. Quantazation requiers GPU though. You'll have to download already quantazed models
from spmd_util import partition_module                # You could experiment with using already quantazed models like 4bit/Llama-2-7b-Chat-GPTQ if you're feeling funny
from datasets import Dataset, load_dataset, concatenate_datasets
from dataclasses import dataclass
from tqdm import tqdm

import transformers
import datasets
import pandas as pd
import numpy as np
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
import torch.utils
from sklearn.metrics import roc_auc_score
try:
    
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ.pop('TPU_PROCESS_ADDRESSES')
    os.environ.pop('CLOUD_TPU_TASK_ID')
    hf_logging.set_verbosity_error() # It can still display warnings which is a bit annoying but whatever
except:
    pass


MAX_INPUT=2048
MODEL = "Locutusque/Hercules-2.5-Mistral-7B" #You should be able to use 7B model with no changes! There should be enough HBM
SAVED_MODEL = "Locutusque/Hercules-2.5-Mistral-7B"


# In[ ]:


class ConversationDataset(TorchDataset):
    def __init__(self, tokenizer, max_length=512, dataset=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        messages = self.dataset[idx]["conversations"]
        text = ""
        for message in messages:
            role = message["from"]
            if role == "system":
                text += f"<|im_start|>system\n{message['value']}<|im_end|>\n"
            if role == "human":
                text += f"<|im_start|>user\n{message['value']}<|im_end|>\n"
            if role == "function-call":
                text += f"<|im_start|>call\n{message['value']}<|im_end|>\n"
            if role == "function-response":
                text += f"<|im_start|>function\n{message['value']}<|im_end|>\n"
            if role =="gpt":
                text += f"<|im_start|>assistant\n{message['value']}{self.tokenizer.eos_token}"
        input_ids = self.tokenizer(text, add_special_tokens=True, max_length=self.max_length, truncation=True, padding="max_length", return_attention_mask=True, return_tensors="pt")
        return {
            "input_ids": input_ids["input_ids"].squeeze(0),
            "labels": input_ids["input_ids"].squeeze(0),
        }


# In[ ]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token


# In[ ]:


train_data = load_dataset("Locutusque/hercules-v2.5", split="train[200000:500000]")
val = load_dataset('Locutusque/hercules-v2.5', split="train[:100]")


# In[ ]:


model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)


# In[ ]:


FLAGS = {'MAX_INPUT': 512,
         'LOGGING_STEPS': 1,
         'NUM_EPOCHS': 1,
         'BATCH_SIZE': 8, #Making batch_size lower then 8 will result in slower training, but will allow for larger models\context. Fortunately, we have 128GBs. Setting higher batch_size doesn't seem to improve time.
          'NUM_STEPS': len(train_data)} 


# In[ ]:


train_data = ConversationDataset(tokenizer, dataset=train_data, max_length=512)
val = ConversationDataset(tokenizer, dataset=val)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data, num_replicas=8, rank=xm.get_ordinal(), shuffle=True)
training_loader = torch.utils.data.DataLoader(train_data, batch_size=FLAGS["BATCH_SIZE"], sampler=train_sampler)
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val, num_replicas=8, rank=xm.get_ordinal(), shuffle=True)
testing_loader = torch.utils.data.DataLoader(val, batch_size=FLAGS["BATCH_SIZE"], sampler=val_sampler)

device = xm.xla_device()


# In[ ]:


def get_nb_trainable_parameters(model):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = get_nb_trainable_parameters(model)

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

cnt = 0
for param in model.parameters():
    cnt += 1
    param.requires_grad = False
    if cnt > 0: # You can set this to a higher value to freeze parameters if your running out of memory.
        param.requires_grad = True
print_trainable_parameters(model)
config = AutoConfig.from_pretrained(MODEL)
num_devices = xr.global_runtime_device_count()
mesh_shape = (1, num_devices, 1)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp'))
partition_module(model, mesh) # After this, the model is sharded between cores but still has the same API as if it was on single device. Neat.


# In[ ]:




def train(FLAGS):
    num_iterations = int(FLAGS['NUM_STEPS'] / FLAGS['BATCH_SIZE'] / 8)
    lr = 2e-6
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=FLAGS['NUM_STEPS'] / FLAGS['BATCH_SIZE'] / 8)
    i = 0
    total_loss = 0
    for epoch in range(1, FLAGS['NUM_EPOCHS'] + 1):
        model.train()
        xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
        for step, batch in enumerate(training_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            xs.mark_sharding(input_ids, mesh, (0, 1)) # Sharding inputs
            xs.mark_sharding(labels, mesh, (0, 1))
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss.requires_grad_()
            loss.backward()
            optimizer.step()
            xm.mark_step()
            if (step + 1) % FLAGS['LOGGING_STEPS'] == 0:
                print(f'loss: {loss.item()}, time: {test_utils.now()}, step: {step}')
 
            scheduler.step()
            i += 1

        model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for step, batch in enumerate(testing_loader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                xs.mark_sharding(input_ids, mesh, (0, 1))
                xs.mark_sharding(labels, mesh, (0, 1))
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1

        average_loss = total_loss / total_steps
        xm.master_print('Epoch {} test end {}, test loss={:.2f}'.format(epoch, test_utils.now(), average_loss))
        xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))


# In[ ]:


train(FLAGS) # I haven't tested the evaluation part in this notebook so hopefully it works. It really should


# In[ ]:


from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("hf_write") # Provide your own HF API token with write access
login(hf_token)
model = model.cpu()
print('now saving the model')
model.push_to_hub(
    SAVED_MODEL, 
    tokenizer=tokenizer,
    safe_serialization=True,
    private=True,
    create_pr=True,
    max_shard_size="2GB", # Sharding isn't as important as before since hardware is better now but who cares anyway
    )# We have to push the model to HF since there is not enough memory on disk. Download weights from there
tokenizer.push_to_hub(
    SAVED_MODEL,
    private=True, 
    
    )

