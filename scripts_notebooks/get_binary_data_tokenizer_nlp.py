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


tokenizer_nlp = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
print('no whitespace')
tokenizer_nlp.pre_tokenizer = Whitespace()
files = ['../TXT_Files/tokens1to25k_noast.txt']
tokenizer_nlp.train(files, trainer)

# reading the data from the file for ast tokenizer

tokenizer_nlp.add_special_tokens(['<pad>',
    '<s>',
    '</s>',
    '<unk>',
    '<mask>',])

tokenizer_nlp.save('tokenizer_nlp')


fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_nlp)
fast_tokenizer.mask_token='<mask>'
fast_tokenizer.pad_token='<pad>'

dataset_codeSearch_python=load_dataset("code_search_net","python")
dpython=dataset_codeSearch_python

class CustomDatasetNonAST(Dataset):
    def __init__(self, tokenizer, evaluate: bool = False, max_encoded_tokens=512):

        self.tokenizer=tokenizer
        self.examples = []
        self.sample_size=18000 # change it to something meaningful later

        self.src_files = dpython
        self.evaluate=evaluate
        
        if self.evaluate:
            for i in range(self.sample_size+1, (self.sample_size+2000),1):#2001,201
                sentences=dpython['train']['whole_func_string'][i]
                encode = self.tokenizer.encode(sentences)
                self.examples += [encode.ids[:max_encoded_tokens]]
                
                if (i%200)==0:
                    print(i,' ')
        else:
            print('-----------------------------')
            for i in range(0,self.sample_size,1):
                sentences=dpython['train']['whole_func_string'][i]
                encode = self.tokenizer.encode(sentences)
                self.examples += [encode.ids[:max_encoded_tokens]]


                if(i%200)==0:
                    print(i,' ')
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i], dtype=torch.int64)

d=CustomDatasetNonAST(tokenizer_nlp, evaluate=False)
e=CustomDatasetNonAST(tokenizer_nlp, evaluate=True)

# Save instance 'd'
with open('../saved_data/noast_d_18k_512_w.pkl', 'wb') as f:
    pickle.dump(d, f)

# Save instance 'e'
with open('../saved_data/noast_e_2k_512_w.pkl', 'wb') as f:
    pickle.dump(e, f)


