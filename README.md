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
```

## notes
- Simple parsing instructions modifications required to apply for other programming languages
- `ast_vocabulary_building.ipynb` has generates the vocabulary that normal nlp tokenizers do not learn at all
- `ast_vocabulary_building.ipynb` has to be modified slightly for other programming languages
- ```

  ```
