import torch

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # refers to a vocabulary of 50,257 words, as used by the BPE tokenizer
    "context_length": 1024,  # maximum number of input tokens the model can handle via the positional embeddings
    "emb_dim": 768,          # represents the embedding size, transforming each token into a 768-dimensional vector.
    "n_heads": 12,           # indicates the count of attention heads in the multi-head attention mechanism
    "n_layers": 12,          # specifies the number of transformer blocks in the model
    "drop_rate": 0.1,        # indicates the intensity of the dropout mechanism (0.1 implies a 10% random drop out of hidden units) to prevent overfitting
    "qkv_bias": False        # determines whether to include a bias vector in the Linear layers of the multi-head attention
}


class DummyGPTModel(torch.nn.Module):

    def __init__(self, config):
        super().__init_()
        self.tok_emb = torch.nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = torch.nn.Embedding(config['context_length'], config['emb_dim'])
        self.drop_emb = torch.nn.Dropout(config['drop_rate'])

        self.trf_blocks = torch.nn.Sequential(*[DummyTransformerBlock(config) for _ in range(config['n_layers'])])
        self.final_norm = DummyLayerNorm(config['emb_dim'])
        self.out_head = torch.nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # WTF is x??
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

# ending before figure 4.1