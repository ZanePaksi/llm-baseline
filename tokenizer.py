import re

class SimpleTokenizerV1:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preproc = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preproc = [item.strip() for item in preproc if item.strip()]
        ids = [self.str_to_int[string] for string in preproc]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:

    """
        This tokenizer handles words that aren't in its vocab by replacing them with <|unk|> and
        adds <|endoftext|> to vocab to separate context between texts
    """

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preproc = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preproc = [item.strip() for item in preproc if item.strip()]
        preproc = [item if item in self.str_to_int else "<|unk|>" for item in preproc]
        ids = [self.str_to_int[string] for string in preproc]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text