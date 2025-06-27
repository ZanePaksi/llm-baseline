"""
    Chapter 4
"""

import torch
import tiktoken
from torch.utils.backcompat import keepdim_warning
import matplotlib.pyplot as plt

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
        super().__init__()
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


def gpt2_start_batch():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print(f"{logits.shape=}")
    print(f"{logits}")


def layer_norm_example():
    torch.manual_seed(123)
    batch = torch.randn(2, 5)
    layer = torch.nn.Sequential(torch.nn.Linear(5, 6), torch.nn.ReLU())
    out = layer(batch)
    print(out)

    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print(f"{mean=}")
    print(f"{var=}")

    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print(f"\nNormalized Outputs\n{out_norm}")
    print(f"Mean: {mean}")
    print(f"Variance: {var}")

    torch.set_printoptions(sci_mode=False)
    print("Mean:\n", mean)
    print("Variance:\n", var)


class LayerNorm(torch.nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, input_tensor):
        mean = input_tensor.mean(dim=-1, keepdim=True)
        var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        norm = (input_tensor - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift


def layer_norm_implementation():
    torch.set_printoptions(sci_mode=False)
    torch.manual_seed(123)
    batch = torch.randn(2, 5)
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
    print(f"{mean=}")
    print(f"{var=}")

# 4.3 GELU

class GELU(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return 0.5 * input_tensor * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                                                       (input_tensor + 0.044715 * torch.pow(input_tensor, 3))))


def gelu_relu_comp():
    gelu, relu = GELU(), torch.nn.ReLU()

    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8, 3))

    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.show()


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


def feed_fwd():
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)

# Stopping at 4.4

class ExampleDeepNeuralNetwork(torch.nn.Module):

    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, input_tensor):
        for layer in self.layers:
            layer_output = layer(input_tensor)
            if self.use_shortcut and input_tensor.shape == layer_output.shape:
                input_tensor = input_tensor + layer_output
            else:
                input_tensor = layer_output
        return input_tensor


def print_gradients(model, input_tensor):
    output = model(input_tensor)
    target = torch.tensor([[0.]])

    loss = torch.nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


def deep_neural_test():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print_gradients(model_without_shortcut, sample_input)
    print()

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, sample_input)




