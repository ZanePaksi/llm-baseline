import torch

from dataset import create_dataloader_v1
from test_dataloader_v1 import data_loader

"""
    This marks the start of figure 2.7
    
    This is a representation of the embedding process
    
    IE: We're converting the tokenized text into tensors (big vectors)
"""

vocab_size = 6
output_dim = 3

input_ids = torch.tensor([2, 3, 5, 1])
# torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(embedding_layer.weight)
# print()

"""
    Remember we embedded 2, 3, 5, 1. We're retrieving the tensor associated with 3
    The embedding_layer var serves as a lookup operation
"""
# print(embedding_layer(torch.tensor([3])))
# print()
#
# print(embedding_layer(input_ids))
# print()

"""
    Just after fig. 2.8, We're going to embed the entire BPE vocab size
    and limit the tensor to 256 dimensions
"""
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

def big_embed():
    max_length = 4
    data_loader = create_dataloader_v1()


