# ast_aware_tokenization
combines ast aware tokenization with subword tokenization(here experimented with BPE with delimeters). Any nlp based toknization can be augmented along with the ast aware tokenization simply by changing couple of lines.

## Usage

`astTokenizer.py` is the file that combines ast aware tokenization. Could be simply imported and used directly for python code.
It uses `tree-sitter` to generate `ast`, then use intelligent parsing to figure out **'ast tokens'**, **'docstrings'**, **'comments'**
Then encode them. For encoding, empty, untrained `huggingface` tokenizer object needs to be passed, that has an **augmented vocabulary**
this allows us to use this as import and in conjuction with `hugginface transformers` library for better integration.


```
all_tokens = list(tokenizer_nlp.get_vocab().keys()) +  list(tokenizer_ast.get_vocab().keys()) #could be done differently
tokenizer_combined = Tokenizer(models.BPE()) # empty Tokenizer object passed
tokenizer_combined.add_tokens(list(set(all_tokens))) 
tokenizer_combined.add_special_tokens(['<pad>',
    '<s>',
    '</s>',
    '<unk>',
    '<mask>',' ','\n'])

custom_t=CustomTokenizer(tokenizer_combined)

ids = custom_t.encode(some_string) # returns the encoded Ids

```
-> Other important utility function have been added to perform encode, decode, inspecting tokens

here we introduced `CustomTokenizer` class that takes the huggingface tokenizer object, and overwrites the `encode` function.
we introduce the ast awareness in this encode function, when some `comments/docstrings/unknkown tokens` of ast vocabulary encountered,
`tokenize_nlp` is called to use simple subword tokenization, but any other natural languagage based tokenization can be applied.
For training purposes `CustomTokenizer` class can be directly used with `PreTrainedTokenzierFast` of `huggingface`

```
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=custom_t)
fast_tokenizer.mask_token='<mask>'
fast_tokenizer.pad_token='<pad>'

# carry out normal transformer model training, might require some additional changes for complex cases

```

### ast aware vocabulary augmentation
- create a file containing tokens, building a ast tokenbase to extract vocabulary from
- `genAstSplits` from `astTokenizer` can be used to create the tokenbase (we used 25k)
- selected 20 most frequent from 3.3 million ast tokens
- this hugely improves the performance of the proposed tokenizer

## Necessary changes for other programming languages
- Simple parsing instructions modifications required to apply for other programming languages
- `ast_vocabulary_building.ipynb` has generates the vocabulary that normal nlp tokenizers do not learn at all
- `ast_vocabulary_building.ipynb` has to be modified slightly for other programming languages
- use tree-sitter build to generate language specific file, change the folder name for parser
- provide tokens parsed file for that specific programming language.

## saved Models
contains the trained transformer models
1. ast_transformer :  contains trained model for proposed tokenizer
2. non_ast_transformer: contains trained model for tokenizer_nlp

## GetEmbed
- generates the embedding for 2 kinds of transformers
- used max_lenth = 512 for tokens
- `compare.ipynb` contains the results
- `all.json` is the SCD-88 python dataset, where train, test, validation combined, as we only need clones
  to check embedding distances

## saved_data

-contains binary files used, can be used to quickly load into the transformer modeln(18k train, 2k eval)
-some naming changes requied

## Pypi
future work

