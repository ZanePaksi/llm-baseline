

# Disclaimer - Learning From a Book

> Build a Large Language Model (From Scratch)  
> By Sebastian Raschka  
> [Link to eBook on Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch)


This is a repository used to store my written examples from a book. I have made small adjustments as continue through the  
text. This code should not be treated as source material, nor a finished product with working examples. Much of this is  
isolated code samples that I have typed and modified from the source material. 

## Notes

General Config File Structure
```
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # refers to a vocabulary of 50,257 words, as used by the BPE tokenizer
    "context_size": 256,  # maximum number of input tokens the model can handle via the positional embeddings
    "emb_dim": 768,          # represents the embedding size, transforming each token into a 768-dimensional vector.
    "n_heads": 12,           # indicates the count of attention heads in the multi-head attention mechanism
    "n_layers": 12,          # specifies the number of transformer blocks in the model
    "drop_rate": 0.1,        # indicates the intensity of the dropout mechanism (0.1 implies a 10% random drop out of hidden units) to prevent overfitting
    "qkv_bias": False        # determines whether to include a bias vector in the Linear layers of the multi-head attention
}
```


### Interesting Links
https://pytorch.org/get-started/locally/#windows-pip