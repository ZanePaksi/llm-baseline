import torch
from attention import MultiHeadAttention
from architecture import FeedForward, LayerNorm, GPT_CONFIG_124M

class TransformerBlock(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=config['emb_dim'],
            d_out=config['emb_dim'],
            context_length=config['context_length'],
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


def test_transformer():
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

