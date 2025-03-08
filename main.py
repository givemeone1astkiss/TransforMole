from transformers.models.bert.tokenization_bert import load_vocab

from transformole import config
from transformole.utils import data


train = open(f'{config.DATA_PATH}/moses/train.csv').read().split('\n')
tokenizer = data.SmilesTokenizer(load_vocab=True)
train = tokenizer.encode(train)
print(train["input_ids"][:5])