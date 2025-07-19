from hujo import torch


class GPTModel(torch.nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        # torch.Embedding is a lookup table that stores embeddings of a fixed dictionary size
        self.token_embedding_lookup: torch.Embedding = torch.nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.positional_embedding_lookup: torch.Embedding = torch.nn.Embedding(config['context_size'], config['emb_dim'])
        self.dropout: torch.Dropout = torch.nn.Dropout(config['drop_rate'])

        self.transformer_blocks: torch.Sequential = torch.nn.Sequential(*[TransformerBlock(config) for _ in range(config['n_layers'])])
        self.final_norm: LayerNorm = LayerNorm(config['emb_dim'])
        self.out_head: torch.Linear = torch.nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

    def forward(self, token_ids: torch.Tensor):
        batch_size, seq_len = token_ids.shape
        token_embeddings: torch.Tensor = self.token_embedding_lookup(token_ids)
        positional_embeddings: torch.Tensor = self.positional_embedding_lookup(torch.arange(seq_len, device=token_ids.device))

        hidden_embeddings: torch.Tensor = token_embeddings + positional_embeddings
        hidden_embeddings: torch.Tensor = self.dropout(hidden_embeddings)
        hidden_embeddings: torch.Tensor = self.transformer_blocks(hidden_embeddings)
        hidden_embeddings: torch.Tensor = self.final_norm(hidden_embeddings)
        logits: torch.Tensor = self.out_head(hidden_embeddings)

        return logits


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_in, d_out, context_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_size, context_size), diagonal=1))

    def forward(self, embedded_tokens):
        b, num_tokens, d_in = embedded_tokens.shape

        keys = self.w_key(embedded_tokens)
        queries = self.w_query(embedded_tokens)
        values = self.w_value(embedded_tokens)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # set the dropout values to negative infinity
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # normalize attention tensor with softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class GELU(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return 0.5 * input_tensor * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                                                       (input_tensor + 0.044715 * torch.pow(input_tensor, 3))))


class FeedForward(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config['emb_dim'], 4 * config['emb_dim']),
            GELU(),
            torch.nn.Linear(4 * config['emb_dim'], config['emb_dim'])
        )

    def forward(self, input_tensor):
        return self.layers(input_tensor)


class TransformerBlock(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=config['emb_dim'],
            d_out=config['emb_dim'],
            context_size=config['context_size'],
            num_heads=config['n_heads'],
            dropout=config['drop_rate'],
            qkv_bias=config['qkv_bias']
        )
        self.feedf = FeedForward(config)
        self.norm1 = LayerNorm(config['emb_dim'])
        self.norm2 = LayerNorm(config['emb_dim'])
        self.drop_shortcut = torch.nn.Dropout(config['drop_rate'])

    def forward(self, input_tensor):
        shortcut = input_tensor

        input_tensor = self.norm1(input_tensor)
        input_tensor = self.attn(input_tensor)
        input_tensor = self.drop_shortcut(input_tensor)
        input_tensor = input_tensor + shortcut

        shortcut = input_tensor
        input_tensor = self.norm2(input_tensor)
        input_tensor = self.feedf(input_tensor)
        input_tensor = self.drop_shortcut(input_tensor)
        input_tensor = input_tensor + shortcut

        return input_tensor


class LayerNorm(torch.nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        # TODO: Define what the hell is happening here. I've forgotten - Bad Bad Bad
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, input_tensor):
        mean = input_tensor.mean(dim=-1, keepdim=True)
        var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        norm = (input_tensor - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift