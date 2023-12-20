# askema groundings

### <a href="https://github.com/advisories/GHSA-v68g-wm8c-6x7j/dependabot?query=user:sushmaakoju">Github Dependabot Advisory</a>:
<a href="https://vuldb.com/?id.248381">Vulnerability in huggingface transformers up to 4.35 </a>. Upgrade to greater than 4.35 version in requirements. Current stable version of Huggingface Transformers is: <a href="https://huggingface.co/docs/transformers/main/en/index"> V4.36.1</a>. 

## Dataset 
Dataset was generated using <a href="https://github.com/sushmaakoju/skema-weekly-status/blob/main/askema-groundings/dataset_generation_askema_groundings_annotations.ipynb">dataset_generation_askema_groundings_annotations</a>

## Dataset format
- The input text data format:
  ```
  Text Context C4 C1
  ```
## Approach
### Revised
- The data would be 3 features
  - [CLS] Text
  - [CLS] C1
  - [CLS] C2
- forward([CLS] Text) => get embedding of [CLS]
- forward([CLS] C1) => get embedding of [CLS]
- forward([CLS] C2) => get embedding of [CLS]
- cosine([CLS]_Text, [CLS]_C1) > cosine([CLS]_Text, [CLS]_C2)

## Triplet Loss and Types of Triplets with TripletLosses

$L(r_a, r_p, r_n) = max(0, m + d(r_a, r_p) - d(r_a, r_n))$


From above loss equation we can have three possibilities or three catergory of triplets.
1. easy-triplets: triplets which have loss of 0 i.e

    **$d(ra,rn) > d(ra,rp) + m$**
2. semi-hard triplets: triplets where the negative is not closer to the anchor than the positive, but which still have positive loss i.e

    **$d(ra,rp) < d(ra,rn) < d(ra,rp) + m$**

3. hard-triplets: triplets where the negative is closer to the anchor than the positive, i.e

    **$d(ra,rn) < d(ra,rp)$**

### BERT
- This is based on the tutorial from <a href="https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/">Chris McCormick BERT word Embeddings tutorial </a>
- Adapted to work for <a href="https://github.com/allenai/scibert">allenai/scibert </a>

### Configuration
Required: Python 3.11x installed.

```
git clone https://github.com/sushmaakoju/skema-weekly-status.git
cd skema-weekly-status/askema-groundings
virtualenv venv --system-site-packages
source venv/bin/activate
pip3 install -r requirements.txt
```

### Tests
Assuming you have completed configuring env.

```
cd skema-weekly-status/askema-groundings
source venv/bin/activate
python skema-weekly-status/askema-groundings/test_example.py
```

## References:
1. <a href="https://medium.com/@harsh.kumar.cse20/understanding-pairwise-ranking-loss-and-triplet-ranking-loss-8c10073c13da">Understanding pairwise-ranking-loss and triplet-ranking-loss with anchor boxes (Image processing)</a>
2. <a href="https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TripletMarginLossMNIST.ipynb">Example TripletMarginLossMNIST</a>
3. <a href="https://doordash.engineering/2021/09/08/using-twin-neural-networks-to-train-catalog-item-embeddings/">using-twin-neural-networks-to-train-catalog-item-embeddings</a>
4. <a href="https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/">Chris McCormick BERT word Embeddings tutorial </a>
5. <a href="https://discuss.huggingface.co/t/how-to-get-cls-embeddings-from-bertfortokenclassification-model/9276/3"> CLS embeddings from BERT model </a>
6. <a href="https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html">torch.nn.TripletMarginLoss</a>
