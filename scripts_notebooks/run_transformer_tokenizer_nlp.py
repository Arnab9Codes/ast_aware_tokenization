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
import transformers
import tokenizers
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Split
import json
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names,get_dataset_config_names
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import BertConfig, BertLMHeadModel
from transformers import Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer
import pickle
from astTokenizer import CustomTokenizer
from customDataset import CustomDatasetNonAST


tokenizer_nlp = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer_nlp.pre_tokenizer = Whitespace()
files = ['tokens1to25k_noast.txt']
tokenizer_nlp.train(files, trainer)

# reading the data from the file for ast tokenizer

tokenizer_nlp.add_special_tokens(['<pad>',
    '<s>',
    '</s>',
    '<unk>',
    '<mask>',' ','\n'])


fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_nlp)
fast_tokenizer.mask_token='<mask>'
fast_tokenizer.pad_token='<pad>'

# Load instance 'd'
print('loading data in d')
#with open('noast_d_18k_embed_512.pkl', 'rb') as f:
with open('noast_d_18k512.pkl', 'rb') as f:
    d_loaded = pickle.load(f)

# Load instance 'e'
print('loading data in e')
#with open('noast_e_2k_embed_512.pkl', 'rb') as f:
with open('noast_e_2k512.pkl', 'rb') as f:
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
    output_dir="../saved_model/non_ast_transformer",
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

