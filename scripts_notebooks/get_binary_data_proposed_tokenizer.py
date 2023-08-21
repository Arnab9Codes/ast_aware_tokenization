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


tokenizer_nlp = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer_nlp.pre_tokenizer = Whitespace()
files = ['./TXT_Files/tokens1to25k_noast.txt']
tokenizer_nlp.train(files, trainer)

# reading the data from the file for ast tokenizer
with open('./TXT_Files/ast_augmented.txt') as file:
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
tokenizer_combined.save('combined_tokenizer')

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=custom_t)
fast_tokenizer.mask_token='<mask>'
fast_tokenizer.pad_token='<pad>'

dataset_codeSearch_python=load_dataset("code_search_net","python")
dpython=dataset_codeSearch_python

class CustomDataset(Dataset):
    def __init__(self, tokenizer, evaluate: bool = False, max_encoded_tokens=512):

        self.custom_tokenizer=tokenizer
        self.examples = []
        self.sample_size=18000 # change it to something meaningful later

        self.src_files = dpython
        self.evaluate=evaluate
        
        if self.evaluate:
            for i in range(self.sample_size+1, (self.sample_size+2000),1):#2001,201
                sentences=dpython['train']['whole_func_string'][i]
                self.examples += [self.custom_tokenizer.encode(sentences)[:max_encoded_tokens]]
                #print("self", self.evaluate+1)
                if (i%200)==0:
                    print(i,' ')
        else:
            print('-----------------------------')
            for i in range(0,self.sample_size,1):
                sentences=dpython['train']['whole_func_string'][i]
                self.examples += [self.custom_tokenizer.encode(sentences)[:max_encoded_tokens]]
                #print(self.examples)
                if(i%200)==0:
                    print(i,' ')
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i], dtype=torch.int64)

d=CustomDataset(custom_t, evaluate=False)
e=CustomDataset(custom_t, evaluate=True)

# Save instance 'd'
with open('../saved_data/dataset_d_18k.pkl', 'wb') as f:
    pickle.dump(d, f)

# Save instance 'e'
with open('../saved_data/dataset_e_2k.pkl', 'wb') as f:
    pickle.dump(e, f)


