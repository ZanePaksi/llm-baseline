import torch


def get_bpe_embedding_layer(vocab_size: int, output_dim: int):
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    return token_embedding_layer


def get_abs_embedding_layer(context_length: int, output_dim: int):
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    return pos_embeddings