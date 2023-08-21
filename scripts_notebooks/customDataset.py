from datasets import load_dataset
import pathlib

import torch
from torch.utils.data.dataset import Dataset
from astTokenizer import CustomTokenizer

dataset_codeSearch_python=load_dataset("code_search_net","python")
dpython=dataset_codeSearch_python


class CustomDataset(Dataset):
    def __init__(self, tokenizer, evaluate: bool = False, max_encoded_tokens=512):

        self.custom_tokenizer=tokenizer
        self.examples = []
        self.sample_size=18000 # 

        self.src_files = dpython
        self.evaluate=evaluate
        
        if self.evaluate:
            for i in range(self.sample_size+1, (self.sample_size+2000),1):#2001,201
                sentences=dpython['train']['whole_func_string'][i]
                self.examples += [self.custom_tokenizer.encode(sentences)[:max_encoded_tokens]]
                #print("self", self.evaluate+1)
        else:
            for i in range(0,self.sample_size,1):
                sentences=dpython['train']['whole_func_string'][i]
                self.examples += [self.custom_tokenizer.encode(sentences)[:max_encoded_tokens]]
                #print(self.examples)
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        return torch.tensor(self.examples[i], dtype=torch.int64)

#d=CustomDataset(custom_t, evaluate=False)
#e=CustomDataset(custom_t, evaluate=True)

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

                #print(self.examples)

                if(i%200)==0:
                    print(i,' ')
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i], dtype=torch.int64)

        
