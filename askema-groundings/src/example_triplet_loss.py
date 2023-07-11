# -*- coding: utf-8 -*-
"""example-triplet-loss-with-unittests.ipynb

Original file is located at
    https://colab.research.google.com/drive/1u9oQBMTRvoF_v-vvwjf55KkPNNKqehu2

## Scibert
- This is based on the tutorial from <a href="https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/">Chris McCormick BERT word Embeddings tutorial </a>
- Adapted to work for <a href="https://github.com/allenai/scibert">allenai/scibert </a>
- The input text data format:

  ```
  Text Context C4 C1
  ```
- Enrique’s comments, guidelines:
    - https://huggingface.co/allenai/scibert_scivocab_uncased
    - https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    - Evaluation scores
    - Later Ensemble models, train the ranker
- Mihai’s guidelines
    - Models produce an embeddings for text : we just use text with 2 candidates (G1 and G2) -> 3 embeddings
    - Embedding(Text)
    - Embedding(G1)
    - Embedding(G2)
    - cosine(Embedding(Text), Embedding(G1)) > cosine(Embedding(Text), Embedding(G2)
    - Forward pass embedding

##### Revised algorithm
- The data would be 3 features
  - [CLS] Text
  - [CLS] C1
  - [CLS] C2
- forward([CLS] Text) => get embedding of [CLS]
- forward([CLS] C1) => get embedding of [CLS]
- forward([CLS] C2) => get embedding of [CLS]
- cosine([CLS]_Text, [CLS]_C1) > cosine([CLS]_Text, [CLS]_C2)how

##### Example
This notebook only includes one example of the format
```
Text Context C4 C1
```

## Citation
Chris McCormick and Nick Ryan. (2019, May 14). *BERT Word Embeddings Tutorial*. Retrieved from http://www.mccormickml.com

## Installation and preprocessing
"""

import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from src.skema_utils import get_cls_embeddings, get_this_triplet_loss, get_triplet_embeddings, init, tokenizer, model
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


"""### Types of Triplets with TripletLosses:
Reference: [triplet-ranking-loss](https://medium.com/@harsh.kumar.cse20/understanding-pairwise-ranking-loss-and-triplet-ranking-loss-8c10073c13da)

Triplet Loss:

$L(r_a, r_p, r_n) = max(0, m + d(r_a, r_p) - d(r_a, r_n))$


From above loss equation we can have three possibilities or three catergory of triplets.
1. easy-triplets: triplets which have loss of 0 i.e

    **d(ra,rn) > d(ra,rp) + m.**
2. semi-hard triplets: triplets where the negative is not closer to the anchor than the positive, but which still have positive loss i.e

    **d(ra,rp) < d(ra,rn) < d(ra,rp) + m.**

3. hard-triplets: triplets where the negative is closer to the anchor than the positive, i.e

    **d(ra,rn) < d(ra,rp).**
"""

def get_triplet_loss_reults(df):
  """ Get triplet loss for this Dataset"""

  triplet_loss_dict = []
  easy_triplets = []
  semi_hard_triplets = []
  hard_triplets = []
  margin = 1.0
  counter = 1

  for i, row in df.iterrows():

    counter += 1
    text = tokenizer.cls_token +" "+ df.at[i,'Text']
    context = tokenizer.cls_token+" "+df.at[i,"Context"]
    c1 = tokenizer.cls_token+" "+df.at[i,"1"]
    c2 = tokenizer.cls_token+" "+df.at[i,"2"]
    text_cls_embeddings, c1_cls_embeddings, c2_cls_embeddings = get_triplet_embeddings(tokenizer, model, text, c1, c2, False)
    this_triplet_loss = get_this_triplet_loss(text_cls_embeddings, c1_cls_embeddings, c2_cls_embeddings )

    positive_distance = (text_cls_embeddings - c1_cls_embeddings).pow(2).sum().sqrt()
    #torch.cdist(txt_last_hidden_states, c1_last_hidden_states)
    negative_distance = (text_cls_embeddings - c2_cls_embeddings).pow(2).sum().sqrt()
    distance_with_margin = torch.add(positive_distance, margin)

    # easy triplet
    if torch.gt(negative_distance, distance_with_margin):
      if counter <=50:
        print("Easy Triplet : ", df.at[i,'Text'], df.at[i,'1'], df.at[i,'2'],this_triplet_loss.item() )
      #print(torch.gt(negative_distance, distance_with_margin))
      easy_triplets.append({"text":df.at[i,'Text'], "c1":df.at[i,'1'], "c2":df.at[i,'2'], "triplet_loss":this_triplet_loss.item()})
    # semi-hard triplet
    elif torch.lt(positive_distance, negative_distance) and torch.lt(negative_distance, distance_with_margin):
      if counter <=50:
        print("Semi-hard Triplet : ", df.at[i,'Text'], df.at[i,'1'], df.at[i,'2'],this_triplet_loss.item() )
      #print(torch.lt(positive_distance, negative_distance) ,  torch.lt(negative_distance, distance_with_margin))
      semi_hard_triplets.append({"text":df.at[i,'Text'], "c1":df.at[i,'1'], "c2":df.at[i,'2'], "triplet_loss":this_triplet_loss.item()})

    # hard-triplet
    elif torch.lt(negative_distance, positive_distance):
      if counter <=50:
        print("Hard Triplet : ", df.at[i,'Text'], df.at[i,'1'], df.at[i,'2'],this_triplet_loss.item() )
      #print(torch.lt(positive_distance, negative_distance))
      hard_triplets.append({"text":df.at[i,'Text'], "c1":df.at[i,'1'], "c2":df.at[i,'2'], "triplet_loss":this_triplet_loss.item()})

    triplet_loss_dict.append({"text":df.at[i,'Text'], "c1":df.at[i,'1'], "c2":df.at[i,'2'], "triplet_loss":this_triplet_loss.item()})
  return triplet_loss_dict, easy_triplets, semi_hard_triplets, hard_triplets

def start(path, files):
    for file in files:
      df = pd.read_csv(file)
      triplet_loss_dict, easy_triplets, semi_hard_triplets, hard_triplets = get_triplet_loss_reults(df[:200])
      print("Total number of easy, semi-hard and hard triplets respectively for this dataset are : %s \n, %s, %s, %s, %s"
                 %(file, str(len(triplet_loss_dict)),str(len(easy_triplets)),str(len(semi_hard_triplets)),str(len(hard_triplets)) ))
      pd.DataFrame.from_dict(triplet_loss_dict).to_csv(os.path.join(path, "triplet_loss_"+os.path.basename(file)[:-4]+".csv"))
      pd.DataFrame.from_dict(easy_triplets).to_csv(os.path.join( path, "easy_triplets_"+os.path.basename(file)[:-4]+".csv"))
      pd.DataFrame.from_dict(semi_hard_triplets).to_csv(os.path.join( path, "semi_hard_triplets_"+os.path.basename(file)[:-4]+".csv"))
      pd.DataFrame.from_dict(hard_triplets).to_csv(os.path.join( path, "hard_triplets_"+os.path.basename(file)[:-4]+".csv"))


if __name__ == '__main__':
  path = input("Enter the path to generated ASKEMA groundings dataset csv files: ")
  print(path)
  if not os.path.exists(path):
    print("Path %s does not exists! Please enter correct path and try again:"%path)
    path = input("Enter the path to generated ASKEMA groundings dataset csv files: ")
    assert os.path.exists(path), "Path %s does not exists! Please enter correct path and try again:"%path
    print(path)

  files = [os.path.join( path, file) for file in os.listdir(path) if file.startswith("ged_5Feb") and ".csv" in file]
  start(path, files)