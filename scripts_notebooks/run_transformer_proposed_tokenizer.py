from tokenizers import Tokenizer, pre_tokenizers
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast
import ast
import json
import astunparse
import tree_sitter # parsing library
from tree_sitter import Language, Parser

from datasets import load_dataset
import pathlib

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import BertConfig, BertLMHeadModel
from transformers import Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer
import pickle
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from astTokenizer import CustomTokenizer
from customDataset import CustomDataset


tokenizer_nlp = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer_nlp.pre_tokenizer = Whitespace()
files = ['./TXT_Files/tokens1to25k_noast.txt']
tokenizer_nlp.train(files, trainer)

# reading the data from the file for ast tokenizer
with open('bpesTxt.txt') as file:
    data = file.read()
  
print("Data type before reconstruction : ", type(data))
      
# reconstructing the data as a dictionary
d = ast.literal_eval(data)
tokenizer_ast = Tokenizer(models.BPE())
tokenizer_ast.add_tokens(list(d.keys()))
all_tokens = list()
all_tokens = list(tokenizer_nlp.get_vocab().keys()) +  list(tokenizer_ast.get_vocab().keys())
tokenizer_combined = Tokenizer(models.BPE())
tokenizer_combined.add_tokens(list(set(all_tokens))) # all_tokens is a list initially as input
tokenizer_combined.add_special_tokens(['<pad>',
    '<s>',
    '</s>',
    '<unk>',
    '<mask>',' ','\n'])

custom_t=CustomTokenizer(tokenizer_combined)

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=custom_t)
fast_tokenizer.mask_token='<mask>'
fast_tokenizer.pad_token='<pad>'

# Load instance 'd'
with open('../saved_data/dataset_d_15k.pkl', 'rb') as f:
    d_loaded = pickle.load(f)

# Load instance 'e'
with open('../saved_data/dataset_e_15k.pkl', 'rb') as f:
    e_loaded = pickle.load(f)

data_collator = DataCollatorForLanguageModeling(
    tokenizer= fast_tokenizer , mlm=True, mlm_probability=0.15, return_tensors='pt'
)

config = BertConfig(50000, hidden_size=30,
                    num_hidden_layers=2, num_attention_heads=2, is_decoder=True,
                    add_cross_attention=True,
                    max_position_embeddings =512
                    )
model = BertLMHeadModel(config)

import torch

training_args = TrainingArguments(
    output_dir="../saved_model/ast_transformer",
    overwrite_output_dir=True,
    num_train_epochs=100,#100,
    do_train=True,
    per_gpu_train_batch_size=128,
    save_steps=200,
    save_total_limit=50,
    logging_steps =100,
    eval_steps=500,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=d_loaded,
    eval_dataset=e_loaded,
    data_collator=data_collator
)

trainer.train()
print(trainer.state.log_history)

