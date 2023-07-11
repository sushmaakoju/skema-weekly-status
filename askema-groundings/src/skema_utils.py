
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import numpy as np
import os
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
# %matplotlib inline

import accelerate
from accelerate import Accelerator

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
model = AutoModelForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states = True,
                                              use_cache=True,# Whether the model returns all hidden-states.
                                    low_cpu_mem_usage=True, offload_state_dict=True)
def init():
  accelerator = Accelerator()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device = accelerator.device
  path = os.path.dirname(os.path.dirname(__file__))
  CACHE_DIR=os.path.join(path, 'transformers-cache')

  # Load pre-trained model tokenizer (vocabulary)
  # tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
  tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)

  # tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
  # model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

  # Load pre-trained model (weights)
  # model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased',
  #                                   output_hidden_states = True,  use_cache=True,# Whether the model returns all hidden-states.
  #                                   cache_dir=CACHE_DIR, low_cpu_mem_usage=True, offload_state_dict=True )
  model = AutoModelForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states = True,
                                              use_cache=True,# Whether the model returns all hidden-states.
                                    cache_dir=CACHE_DIR, low_cpu_mem_usage=True, offload_state_dict=True)
  import tempfile

  offload_dir='/content/offload'
  os.makedirs(offload_dir) if not os.path.exists(offload_dir) else None

  with tempfile.TemporaryDirectory() as tmp_dir:
      model.save_pretrained(tmp_dir, max_shard_size="200MB")
      print('Temp Dir Path:', tmp_dir)
      print(sorted(os.listdir(tmp_dir)))
      # model = BertModel.from_pretrained(tmp_dir, low_cpu_mem_usage=True,offload_folder=offload_dir,
      #                               output_hidden_states = True,  use_cache=True,# Whether the model returns all hidden-states.
      #                               cache_dir=CACHE_DIR, offload_state_dict=True )
      model = AutoModelForMaskedLM.from_pretrained(tmp_dir, low_cpu_mem_usage=True,offload_folder=offload_dir,
                                    output_hidden_states = True, use_cache=True,# Whether the model returns all hidden-states.
                                    cache_dir=CACHE_DIR, offload_state_dict=True)
  return tokenizer, model

def get_cls_embeddings(tokenizer, model, data, is_hidden_states_embeddings = False):
  inputs = tokenizer(data, return_tensors = 'pt', padding=True)
  data_outputs = model(**inputs, output_hidden_states=True)
  data_last_hidden_states = data_outputs.hidden_states[-1]
  data_cls_embeddings = data_outputs.hidden_states[-1][:,0,:]
  # .detach().numpy()
  if is_hidden_states_embeddings:
    return data_last_hidden_states
  return data_cls_embeddings

def get_triplet_embeddings(tokenizer, model, texts, c1, c2, is_hidden_states_embeddings = False):
  # text = tokenizer.cls_token +" "+ df.at[0,'Text']
  # context = tokenizer.cls_token+" "+df.at[0,"Context"]
  # C1 = tokenizer.cls_token+" "+df.at[0,"1"]
  # C2 = tokenizer.cls_token+" "+df.at[0,"2"]
  text_cls_embeddings = get_cls_embeddings(tokenizer, model, texts, is_hidden_states_embeddings)
  c1_cls_embeddings = get_cls_embeddings(tokenizer, model, c1, is_hidden_states_embeddings)
  c2_cls_embeddings = get_cls_embeddings(tokenizer, model, c2, is_hidden_states_embeddings)
  return text_cls_embeddings, c1_cls_embeddings, c2_cls_embeddings

def get_this_triplet_loss(text_cls_embeddings, c1_cls_embeddings, c2_cls_embeddings ):
  triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
  this_triplet_loss = triplet_loss(text_cls_embeddings, c1_cls_embeddings, c2_cls_embeddings)
  return this_triplet_loss

