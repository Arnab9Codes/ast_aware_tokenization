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
from tree_sitter import Language, Parser

from datasets import load_dataset
import pathlib

import torch
from torch.utils.data.dataset import Dataset

PYTHON_LANGUAGE = Language('build/my-languages.so', 'python')
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)


start_points=[]
end_points=[]


#function for finding the leaf nodes while parsing the AST
def walk(node):
    """
    node: takes ast.root_node to start traversing
    returns-> startpoints: starting index of each token
              endpointes: ending index of each token 
    """
    start_points = []
    end_points = []
    
    stack = [node]
    
    while stack:
        curr_node = stack.pop()

        if(curr_node.type=='expression_statement' and len(curr_node.children)==1 and curr_node.children[0].type=='string'):
            start_points.append(curr_node.start_point)
            end_points.append(curr_node.end_point)
        
        elif (len(curr_node.children)) == 0:
            start_points.append(curr_node.start_point)
            end_points.append(curr_node.end_point)
        else:
            for child in curr_node.children:
                stack.append(child)
    
    start_points.reverse()
    end_points.reverse()
    
    return start_points, end_points

# splitting each code_string by new line to get each line
def break_into_lines(code_sample):
    """
    code_sample: code taken as a single string
    return: each line in the corresponding code
    """
    return code_sample.split('\n')

# functions for getting tokens from a single code string
def get_tokens(code_sample, start_points, end_points, check_tab_new_line=True):
    """
    code_sample: full code in string
    startpoints: starting index of each token
    endpointes: ending index of each token 
    """
    tokens=[]

    lines_in_code=break_into_lines(code_sample)

    assert len(start_points)==len(end_points), 'problem in finding the start and end points in the code'

    num_of_lines=len(lines_in_code)

    old_line = start_points[0][0]
    new_line = start_points[0][0]
    flag_new_line=False
    
    for i in range(len(start_points)):
        new_line=start_points[i][0]
        
        if(check_tab_new_line==True):
            if(new_line > old_line): 
                tokens.append('\n')

                flag_new_line = True
                old_line = start_points[i][0]
                #print("it is in if condition")
            else:
                flag_new_line = False

        if start_points[i][0]==end_points[i][0]:

            if(check_tab_new_line==True):
                if flag_new_line:
                    #for tabs in range(start_points[i][1]):
                    #    tokens.append('   ')
                    # above line is an option but might varry during the interpretation of tab as space in some ide
                    
                    tokens.append(lines_in_code[start_points[i][0]][:start_points[i][1]])
                    # above line is another option but it will have to be taken care of by tokenizer_nlp
            tokens.append(lines_in_code[start_points[i][0]][start_points[i][1]:end_points[i][1]])
            
        else:
            string=''
            string += lines_in_code[start_points[i][0]][start_points[i][1]: len(lines_in_code[start_points[i][0]])]
            string+='\n'
            for j in range(start_points[i][0]+1, end_points[i][0]):
                string+=lines_in_code[j]
                string+='\n'
        
            string += lines_in_code[end_points[i][0]][0:end_points[i][1]]

            tokens.append(string)
        
    return tokens


def getAstSplits(source_code):    

    tree = parser.parse(bytes(source_code, "utf8"))
    start_points_per_file, end_points_per_file = walk(tree.root_node)
    tokens_per_file = get_tokens(source_code, start_points_per_file, end_points_per_file)
    
    return tokens_per_file


class CustomTokenizer:
    ''' 
    can not subclass huggingface class
    need to preinitiate the class before and override functions here
    '''
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokens_ast=[] # only ast
        self.tokens_all=[] # both ast and nlp
        self.vocab = list(self.tokenizer.get_vocab().keys())

    def tokenize_nlp(self, text):
        nlp_tokens = []
        token = None
        for c in text:
            # expand previous token by one character or append previous token to tokens
            if token is not None:
                new_token = token + c
                if new_token not in self.vocab:
                    nlp_tokens.append(token)
                    token = None
                else:
                    token = new_token

            if c not in self.vocab:
                nlp_tokens.append('<unk>')

            # begin new token
            elif token is None:
                token = c
    
        # append last token
        if token:
            nlp_tokens.append(token)
        #print(nlp_tokens)
        return nlp_tokens

    def encode(self, code):
        # encode function returns the ids after tokenzization
        # print("doing not soo much at all.")
        self.tokens_ast = getAstSplits(code)

        encoding=[]
        for token in self.tokens_ast:
            
            try:
                if self.tokenizer.token_to_id(token)!=None:
                    encoding.append(self.tokenizer.token_to_id(token))
                    self.tokens_all.append(token)

                else:# have to modify the code here for natural language based tokenization
                    
                    nlp_tks=self.tokenize_nlp(token)

                    for t in nlp_tks:
                        self.tokens_all.append(t)

                        if (self.tokenizer.token_to_id(t)!=None):
                            encoding.append(self.tokenizer.token_to_id(t))
                        else:
                            encoding.append(self.tokenizer.token_to_id('<unk>'))

            except Exception as e:
                print('token: ',token, "some problem in encoding the code.")

        return encoding

    def encode_nlp(self, nlp_string):
        encoding=[]
        
        return encoding


    def decode(self,encoded_ids):
        tokens_list=[]
        for id in encoded_ids:
            tokens_list.append(self.tokenizer.id_to_token(id))

        return tokens_list
    
    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)
    
    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)
    
    def get_vocab_size(self, with_added_tokens=True):
        return self.tokenizer.get_vocab_size()
    
    def get_tokens(self):
        return self.tokens_all

    def get_tokens_ast(self, ast_only=True):    
        return self.tokens_ast
    
    def save(self, name):
        return self.tokenizer.save(name)
